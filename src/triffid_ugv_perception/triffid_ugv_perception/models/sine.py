"""Official pretrained SINE instance segmentation with cached K-way supports.

Uses inference_fsod's SINE/MFormer unchanged. No base-weight download, target
fine-tuning, or query-to-support feedback. Inputs match its dataset mapper:
ImageNet RGB normalization, long-edge resize, then zero-pad normalized images.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import cv2
import numpy as np
from ..ontology import classes_for
from ..supports import SupportBank
from .devit import file_hash


def suppress_overlapping_masks(detections, iou_threshold):
    """Class-agnostic mask NMS; preserve distinct/adjacent instance masks."""
    kept=[]
    areas={id(d):int(np.count_nonzero(d['mask'])) for d in detections}
    for candidate in sorted(detections,key=lambda d:d['confidence'],reverse=True):
        duplicate=False
        for existing in kept:
            a,b,c,e=map(int,candidate['bbox']);x,y,z,w=map(int,existing['bbox'])
            left,top,right,bottom=max(a,x),max(b,y),min(c,z),min(e,w)
            if right<=left or bottom<=top:continue
            intersection=int(np.count_nonzero(candidate['mask'][top:bottom,left:right] & existing['mask'][top:bottom,left:right]))
            union=areas[id(candidate)]+areas[id(existing)]-intersection
            if union and intersection/union>iou_threshold:
                duplicate=True;break
        if not duplicate:kept.append(candidate)
    return kept


class SineDetector:
    def __init__(self, config):
        import torch
        self.torch = torch
        self.classes = tuple(classes_for(config['classes']))
        self.by_id = {c.id:c for c in self.classes}
        self.selection = SupportBank(config['support_bank']).select(self.classes,config['shots'])
        self.repo = Path(config['sine_repo']).resolve()
        checkpoint = Path(config['sine_checkpoint']).resolve()
        for path in (self.repo/'inference_fsod/fsdet/model/model.py',checkpoint):
            if not path.is_file(): raise ValueError(f'SINE dependency missing: {path}')
        self.device = str(config['device'])
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; select device=cpu explicitly')
        precision=config['sine_precision']
        if precision not in ('float16','float32'):
            raise ValueError('SINE precision must be float16 or float32')
        self.half = precision=='float16' and self.device.startswith('cuda')
        self.precision = 'float16' if self.half else 'float32'
        self.size = int(np.ceil(int(config['image_size'])/14)*14)
        self.score = float(config['confidence_threshold'])
        self.nms_iou = float(config['sine_mask_nms_iou'])
        if not 0<self.nms_iou<=1:raise ValueError('Invalid SINE mask NMS IoU')
        if self.size<28 or not 0<=self.score<=1:
            raise ValueError('Invalid SINE image size or confidence threshold')
        sys.path.insert(0,str(self.repo))
        sys.path.insert(0,str(self.repo/'inference_fsod'))
        from fsdet.model.model import SINE
        from fsdet.model.image_encoder.dinov2 import DINOv2EncoderViT, LayerNorm2d
        from fsdet.model.transofrmer_decoder import mformer
        from fsdet.sine_config import add_sine_config
        from dinov2.models import vision_transformer as vits
        from detectron2.config import get_cfg
        from detectron2.data.transforms import ResizeShortestEdge
        if Path(sys.modules[SINE.__module__].__file__).resolve()!=self.repo/'inference_fsod/fsdet/model/model.py':
            raise RuntimeError('Another fsdet package shadows the requested SINE checkout')
        cfg=get_cfg();add_sine_config(cfg)
        cfg.merge_from_file(str(self.repo/'inference_fsod/configs/Base-SINE.yaml'))
        encoder=vits.vit_large(img_size=518,patch_size=14,init_values=1e-5,
            ffn_layer='mlp',block_chunks=0,qkv_bias=True,proj_bias=True,ffn_bias=True)
        # Older installed Detectron2 lacks get_norm("LN"). SINE already ships
        # the identical channel-wise LayerNorm2d (eps=1e-6); use it only while
        # constructing the published decoder, then restore the dependency hook.
        original_norm=mformer.get_norm
        self.norm_compatibility=False
        try: original_norm('LN',1)
        except KeyError:
            self.norm_compatibility=True
            mformer.get_norm=lambda name,channels:LayerNorm2d(channels) if name=='LN' else original_norm(name,channels)
        try: decoder=mformer.build_mformer(cfg)
        finally: mformer.get_norm=original_norm
        self.model=SINE(image_encoder=DINOv2EncoderViT(encoder,out_chans=cfg.MODEL.DINO.OUT_CHANS),
            transformer_decoder=decoder,preprocess=True,
            sem_seg_postprocess_before_inference=True,num_classes=len(self.classes),
            test_topk_per_image=cfg.MODEL.SINE.test_topk_per_image,
            score_threshold=self.score,use_id_query=cfg.MODEL.SINE.use_id_query)
        state=torch.load(checkpoint,map_location='cpu')
        state=state.get('model',state)
        # The public checkpoint includes the training criterion. It is not part
        # of the official inference model; every encoder/decoder tensor is strict.
        criterion={k for k in state if k.startswith('criterion.')}
        if criterion-{'criterion.empty_weight'}:
            raise ValueError(f'Unexpected SINE training-only weights: {criterion}')
        self.model.load_state_dict({k:v for k,v in state.items() if k not in criterion},strict=True)
        del state,encoder
        self.model.to(device=self.device,dtype=torch.float16 if self.half else torch.float32).eval().requires_grad_(False)
        self.resize=ResizeShortestEdge(self.size,self.size)
        self.checkpoint_hash=file_hash(checkpoint)
        self.revision=subprocess.check_output(['git','-C',str(self.repo),'rev-parse','HEAD'],text=True).strip()
        identity=dict(schema='triffid.sine.supports.v1',supports=self.selection.metadata(),
            checkpoint_sha256=self.checkpoint_hash,source_revision=self.revision,
            image_size=self.size,precision=self.precision,method='official do_preprocess + integrate_queries')
        cache=Path(config['sine_support_cache']).resolve();cache.mkdir(parents=True,exist_ok=True)
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self.cache=cache/(key+'.pth')
        self.query_encodings=0
        if self.cache.is_file():
            data=torch.load(self.cache,map_location='cpu')
            if data['metadata']!=identity:raise ValueError('SINE support cache metadata mismatch')
            for name,tensor in data['buffers'].items():self.model.register_buffer(name,tensor.to(self.device))
        else:
            self._prepare_supports()
            buffers={name:getattr(self.model,name).detach().cpu() for name in ('id_labels','id_queries','seg_labels','seg_queries')}
            fd,temporary=tempfile.mkstemp(dir=cache,prefix='.sine-',suffix='.pth');os.close(fd)
            try:
                torch.save(dict(metadata=identity,buffers=buffers),temporary)
                os.replace(temporary,self.cache)
            finally:
                if os.path.exists(temporary):os.unlink(temporary)
        if set(self.model.seg_labels.cpu().tolist())!=set(self.by_id):
            raise ValueError('SINE cache omits or adds selected class IDs')
        for name in ('id_queries','seg_queries'):
            setattr(self.model,name,getattr(self.model,name).to(dtype=torch.float16 if self.half else torch.float32))
        self.model.preprocess=False
        self.model.ref_feat_dict.clear();self.model.ref_mask_dict.clear()
        self._last_features=None
        self._hook=self.model.image_encoder.neck.register_forward_hook(self._capture_features)

    def _amp(self):
        return self.torch.cuda.amp.autocast(enabled=True,dtype=self.torch.float16) if self.half else contextlib.nullcontext()

    def _capture_features(self,module,inputs,output):
        self._last_features=output.detach()
        self.query_encodings+=1

    def _input(self,bgr):
        torch=self.torch
        from detectron2.structures import Instances
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        transform=self.resize.get_transform(rgb)
        resized=transform.apply_image(rgb)
        tensor=torch.as_tensor(np.ascontiguousarray(resized.transpose(2,0,1))).float()/255
        tensor=(tensor-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        tensor=torch.nn.functional.pad(tensor,(0,self.size-tensor.shape[2],0,self.size-tensor.shape[1]))
        inputs=dict(image=tensor,height=bgr.shape[0],width=bgr.shape[1],instances=Instances(resized.shape[:2]))
        return inputs,transform,resized.shape[:2]

    def _prepare_supports(self):
        torch=self.torch
        grouped={}
        for item in self.classes:
            for support in self.selection.for_class(item):grouped.setdefault(support.image,[]).append((item,support))
        with torch.inference_mode(),self._amp():
            for path,entries in grouped.items():
                rgb=cv2.imread(str(path))
                if rgb is None:raise ValueError(f'Unreadable support: {path}')
                inputs,transform,_=self._input(rgb)
                masks=[];labels=[]
                for item,support in entries:
                    raw=cv2.imread(str(support.mask),cv2.IMREAD_UNCHANGED)
                    binary=raw>0 if support.foreground_value is None else raw==support.foreground_value
                    mask=transform.apply_segmentation(binary.astype(np.uint8))
                    if not mask.any():raise ValueError(f'{support.example_id}: foreground vanished during resizing')
                    masks.append(torch.from_numpy(mask.copy()));labels.append(item.id)
                inputs['instances'].gt_classes=torch.tensor(labels,dtype=torch.int64)
                inputs['instances'].gt_masks=torch.stack(masks)
                self.model([inputs])
        # The official integrator pools the saved reference tensors on CPU.
        # Use float32 here because CPU half-precision einsum is unsupported by
        # the legacy runtime; encoder/decoder inference remains GPU FP16.
        self.model.ref_feat_dict={k:[v.float() for v in values] for k,values in self.model.ref_feat_dict.items()}
        self.model.ref_mask_dict={k:[v.float() for v in values] for k,values in self.model.ref_mask_dict.items()}
        with torch.inference_mode():self.model.integrate_queries()

    def _appearance(self,mask):
        torch=self.torch
        h,w=self._resized_shape
        resized=cv2.resize(mask.astype(np.float32),(w,h),interpolation=cv2.INTER_NEAREST)
        padded=np.zeros((self.size,self.size),np.float32);padded[:h,:w]=resized
        features=self._last_features[0].float()
        weights=torch.nn.functional.interpolate(torch.from_numpy(padded)[None,None].to(features.device),
            size=features.shape[-2:],mode='area')[0,0]
        vector=(features*weights).sum((1,2))/weights.sum().clamp_min(1e-8)
        return torch.nn.functional.normalize(vector,dim=0).cpu().numpy().tolist()

    def _decode(self,instances):
        instances=instances.to('cpu')
        result=[]
        for raw,cls,score in zip(instances.pred_masks.numpy(),instances.pred_classes.numpy(),instances.scores.numpy()):
            cls=int(cls);score=float(score)
            if cls not in self.by_id:raise ValueError(f'SINE returned unselected class ID {cls}')
            if not np.isfinite(score) or score<self.score:continue
            mask=raw>0
            ys,xs=np.where(mask)
            if not len(xs):continue
            item=self.by_id[cls]
            result.append(dict(bbox=(float(xs.min()),float(ys.min()),float(xs.max()+1),float(ys.max()+1)),
                class_id=item.id,class_name=item.name,confidence=score,mask=mask))
        result=suppress_overlapping_masks(result,self.nms_iou)
        for detection in result:detection['appearance']=self._appearance(detection['mask'])
        return result

    def detect(self,bgr):
        inputs,_,self._resized_shape=self._input(bgr)
        try:
            with self.torch.inference_mode(),self._amp():
                output=self.model([inputs])
            return self._decode(output[0]['instances'])
        finally:self._last_features=None

    def metadata(self):
        return dict(name='sine',source_repository='https://github.com/aim-uofa/SINE',source_revision=self.revision,
            checkpoint_sha256=self.checkpoint_hash,ways=len(self.classes),support_bank=self.selection.metadata(),
            support_cache=str(self.cache),image_size=self.size,precision=self.precision,
            instance_method='official cached multi-class SINE instance masks; boxes derived from masks',
            score_type='official class probability, after class-probability times mask-quality threshold',
            mask_nms_iou=self.nms_iou,confidence_threshold=self.score,query_encodings=self.query_encodings,
            detectron2_ln_compatibility=self.norm_compatibility)
