"""Pretrained MASA-R50 with external detections and class-compatible memory."""
from collections import defaultdict, deque
import importlib
from pathlib import Path
import subprocess
import sys
import types

import cv2
import numpy as np


class MasaTracker:
    def __init__(self, memory_seconds=60, capture_candidates=False, class_correction=False, repo=None, checkpoint=None):
        self.capture_candidates=capture_candidates
        self.class_correction=class_correction
        import torch
        from mmengine.config import Config
        from mmdet.utils import register_all_modules
        from mmdet.registry import MODELS
        from ..models.devit import file_hash
        root=Path(__file__).absolute().parents[4]
        repo=Path(repo).resolve() if repo else root/'ugv_models/masa'
        register_all_modules()
        path=repo/'masa/models/tracker/masa_tao_tracker.py'
        source=path.read_text()
        marker='# Respect the advertised category gate for heterogeneous objects.'
        if marker not in source:
            raise RuntimeError('MASA source lacks the validated category-gate patch; prepare the pinned vendor source before startup')
        if class_correction:
            from .class_correction import ensure_masa_hooks
            ensure_masa_hooks(path)
        for name,relative in [('masa','masa'),('masa.models','masa/models'),
                              ('masa.models.necks','masa/models/necks'),('masa.models.mot','masa/models/mot'),
                              ('masa.models.tracker','masa/models/tracker')]:
            module=types.ModuleType(name);module.__path__=[str(repo/relative)];sys.modules[name]=module
        for name in ['masa.models.losses','masa.models.necks.deform_fusion',
                     'masa.models.tracker.masa_tao_tracker','masa.models.mot.masa']:
            importlib.import_module(name)
        cfg=Config.fromfile(str(repo/'configs/masa-one/masa_r50_plug_and_play.py'))
        cfg.model.tracker.update(fps=1,memo_tracklet_frames=memory_seconds,max_distance=-1,
                                with_cats=True,init_score_thr=.35,obj_score_thr=.1)
        self.model=MODELS.build(cfg.model)
        if class_correction:
            from .class_correction import ClassChangeAssociation
            self.model.tracker.class_change_policy=ClassChangeAssociation()
        checkpoint=Path(checkpoint).resolve() if checkpoint else root/'ugv_models/masa_r50.pth'
        self.checkpoint_hash=file_hash(checkpoint)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model.load_state_dict(state.get('state_dict',state),strict=True)
        del state
        self.model.cuda().eval().requires_grad_(False)
        original_predict=self.model.track_head.predict
        def capture(*args,**kwargs):
            result=original_predict(*args,**kwargs)
            self.embeddings=result.detach()
            return result
        self.model.track_head.predict=capture
        self.torch=torch;self.memory_seconds=memory_seconds
        self.revision=subprocess.check_output(['git','-c',f'safe.directory={repo}','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        self.tracker_source_hash=file_hash(path)
        self.first_timestamp=self.last_timestamp=None
        self.calls=0;self.image_encodings=0

    def metadata(self):
        return dict(name='masa_r50',source_revision=self.revision,checkpoint_sha256=self.checkpoint_hash,tracker_source_sha256=self.tracker_source_hash,
                    strict_checkpoint_load=True,memory_seconds=self.memory_seconds,fps=1,
                    class_compatibility='soft_with_guarded_class_changes' if self.class_correction else True,spatial_gate=False,init_score_threshold=.35,
                    recovery_score_threshold=.1,match_score_threshold=.5,image_size=1024,
                    image_encodings=self.image_encodings,appearance='pretrained MASA instance embeddings')

    def update(self,detections,bgr,timestamp_ns=None):
        from mmdet.structures import DetDataSample,TrackDataSample
        torch=self.torch
        timestamp=int(timestamp_ns) if timestamp_ns is not None else self.calls*10**9
        if self.last_timestamp is not None and timestamp<=self.last_timestamp:
            raise ValueError('Tracking timestamps must increase')
        if self.first_timestamp is None:self.first_timestamp=timestamp
        self.last_timestamp=timestamp;self.calls+=1
        # Native MASA resets whenever frame_id==0. Positive subsecond gaps
        # must advance beyond zero too (e.g. a two-image camera smoke test).
        tick=(timestamp-self.first_timestamp+10**9-1)//10**9
        # Upstream expires memory after matching; prune first, including on empty frames.
        self.model.tracker.tracks={k:v for k,v in self.model.tracker.tracks.items()
                                   if tick-v['last_frame']<self.memory_seconds}
        self.last_candidate_observations=[];self.last_raw_embeddings=None
        if not detections:return []
        h,w=bgr.shape[:2];scale=1024/max(h,w);nw,nh=round(w*scale),round(h*scale)
        rgb=cv2.resize(bgr,(nw,nh))[:,:,::-1].copy()
        tensor=torch.from_numpy(rgb.transpose(2,0,1)).float().cuda()
        mean=tensor.new_tensor([123.675,116.28,103.53])[:,None,None]
        std=tensor.new_tensor([58.395,57.12,57.375])[:,None,None]
        tensor=torch.nn.functional.pad((tensor-mean)/std,(0,(-nw)%32,0,(-nh)%32))
        sample=DetDataSample(metainfo=dict(frame_id=int(tick),ori_shape=(h,w),img_shape=(nh,nw),scale_factor=(nw/w,nh/h)))
        sample.det_bboxes=torch.tensor([[*d['bbox'],d['confidence']] for d in detections],device='cuda')
        # MASA reserves negative labels for invalid detections. Our internal
        # physical-person group is -1, so encode it as a non-ontology sentinel.
        sample.det_labels=torch.tensor([10000 if d['class_id']==-1 else d['class_id'] for d in detections],device='cuda')
        sequence=TrackDataSample(video_data_samples=[sample])
        self.embeddings=None
        if getattr(self,'class_correction',False):self.model.tracker.class_change_policy.set_sources(detections)
        with torch.inference_mode():
            output=self.model.predict(tensor[None,None],[sequence])[0][0].pred_track_instances
        self.image_encodings+=1
        vectors=torch.nn.functional.normalize(self.embeddings.float(),dim=1).cpu().numpy().tolist()
        if getattr(self,'capture_candidates',False):
            self.last_candidate_observations=[dict(d,appearance=vectors[i]) for i,d in enumerate(detections)]
            self.last_raw_embeddings=self.embeddings.float().cpu().numpy().tolist()
        def key(box,score,label):
            return (int(label),float(np.float32(score)),tuple(np.asarray(box,dtype=np.float32).tolist()))
        indices=defaultdict(deque)
        for i,d in enumerate(detections):indices[key(d['bbox'],d['confidence'],d['class_id'])].append(i)
        tracks=[]
        for box,score,label,tid in zip(output.bboxes.cpu().tolist(),output.scores.cpu().tolist(),
                                      output.labels.cpu().tolist(),output.instances_id.cpu().tolist()):
            label=-1 if label==10000 else label
            lookup=key(box,score,label)
            if not indices[lookup]:raise RuntimeError('MASA output lost its exact detection source')
            index=indices[lookup].popleft()
            track=dict(detections[index],bbox=box,confidence=score,track_id=int(tid),
                       appearance=vectors[index],recovered=score<.35)
            if getattr(self,'class_correction',False):
                encoded=10000 if label==-1 else label
                track['class_evidence']=self.model.tracker.class_change_policy.evidence.get(key(box,score,encoded),[])
                track['identity_class_id']=-2
            tracks.append(track)
        return tracks
