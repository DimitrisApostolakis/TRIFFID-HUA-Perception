"""YOLOE-v8-L visual supports, matching the evaluated PUI adapter."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import cv2
import numpy as np
from ..ontology import CLASSES, classes_for
from ..supports import SupportBank
from .devit import file_hash

class YoloeDetector:
    def __init__(self,config):
        import torch
        self.torch=torch;self.device=config['device']
        repo=Path(config['yoloe_repo']).resolve();checkpoint=Path(config['yoloe_checkpoint']).resolve()
        sys.path.insert(0,str(repo))
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
        import ultralytics
        if Path(ultralytics.__file__).resolve().parent!=repo/'ultralytics':
            raise ValueError('Another Ultralytics install shadows the evaluated YOLOE checkout')
        bank=SupportBank(config['support_bank'])
        requested=config['classes'].strip()
        if requested:self.classes=tuple(classes_for(requested))
        else:
            ready=[]
            for c in CLASSES:
                # Empty reserved files are explicit unavailable classes.
                if c.name not in bank.data['classes']:continue
                slots=bank.for_class(c)
                if slots[0].image.is_file() and slots[0].mask.is_file():ready.append(c)
            self.classes=tuple(ready)
        self.selection=bank.select_available(self.classes)
        checkpoint_hash=file_hash(checkpoint)
        selection_metadata=self.selection.metadata()
        selection_hash=hashlib.sha256(json.dumps(selection_metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        for expected,actual in [('yoloe_expected_checkpoint_sha256',checkpoint_hash),
                ('support_bank_expected_sha256',self.selection.manifest_sha256),('support_selection_expected_sha256',selection_hash)]:
            if config.get(expected) and config[expected]!=actual:raise ValueError(f'Live asset fingerprint changed: {expected}')
        self.size=int(config['image_size']);self.score=float(config['confidence_threshold'])
        self.floor=float(config['tracking_confidence_floor']);self.nms=float(config['yoloe_nms_iou'])
        if not 0<=self.floor<=self.score<=1 or self.size<32 or not 0<self.nms<=1:
            raise ValueError('Invalid YOLOE resolution or thresholds')
        revision=subprocess.check_output(['git','-c',f'safe.directory={repo}','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if config.get('yoloe_expected_source_revision') and config['yoloe_expected_source_revision']!=revision:
            raise ValueError('Live YOLOE source revision changed')
        identity=dict(schema='triffid.yoloe.visual-supports.v1',source_revision=revision,
            checkpoint_sha256=checkpoint_hash,support_bank=selection_metadata,image_size=self.size,
            support_aggregation='normalized class mean of all selected visual box embeddings',
            prompt_box_policy='tight union of annotated class mask')
        cache=Path(config['yoloe_support_cache']);cache.mkdir(parents=True,exist_ok=True)
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        path=cache/(key+'.pth');hit=path.is_file()
        self.model=YOLOE(str(checkpoint));self.model.to(self.device)
        if hit:
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['metadata']!=identity:raise ValueError('YOLOE cache metadata mismatch')
            memory=saved['memory']
        else:
            vectors=[]
            with torch.inference_mode():
                for c in self.classes:
                    examples=[]
                    for support in self.selection.for_class(c):
                        rgb,mask=support.load();y,x=np.nonzero(mask)
                        box=np.array([[x.min(),y.min(),x.max()+1,y.max()+1]],np.float32)
                        self.model.predict(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),imgsz=self.size,device=self.device,
                            prompts=dict(bboxes=box,cls=np.array([0])),predictor=YOLOEVPSegPredictor,
                            return_vpe=True,verbose=False,save=False,conf=.99)
                        examples.append(self.model.predictor.vpe[0,0].detach())
                    vectors.append(torch.nn.functional.normalize(torch.stack(examples).mean(0),dim=0))
                    print(f'YOLOE live supports: {c.name}: {len(examples)}',flush=True)
            memory=torch.stack(vectors)[None].cpu()
            fd,tmp=tempfile.mkstemp(dir=cache,suffix='.pth');os.close(fd)
            try:
                torch.save(dict(metadata=identity,memory=memory),tmp);os.replace(tmp,path)
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
        if memory.ndim!=3 or memory.shape[:2]!=(1,len(self.classes)) or not torch.isfinite(memory).all():
            raise ValueError('Invalid YOLOE visual support cache')
        self.model.set_classes([c.name for c in self.classes],memory.to(self.device));self.model.predictor=None
        self.info=dict(identity,name='yoloe_v8l',backbone='YOLOE-v8-L',confidence_threshold=self.score,
            support_selection_sha256=selection_hash,
            tracking_confidence_floor=self.floor,nms_iou=self.nms,support_cache=str(path),support_cache_hit=hit,
            precision='float32',text_prompts=False,ultralytics_version=ultralytics.__version__,
            unavailable_classes=[c.name for c in CLASSES if c not in self.classes])
        self.last_tracking_detections=[]

    def detect(self,bgr):
        result=self.model.predict(bgr,imgsz=self.size,device=self.device,conf=self.floor,iou=self.nms,
            verbose=False,save=False,retina_masks=True)[0]
        rows=[]
        for i,(box,score,index) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy())):
            c=self.classes[int(index)]
            mask=result.masks.data[i].cpu().numpy()>.5 if result.masks is not None else None
            if mask is not None and mask.shape!=bgr.shape[:2]:raise ValueError('YOLOE mask has wrong resolution')
            rows.append(dict(bbox=list(map(float,box)),confidence=float(score),class_id=c.id,class_name=c.name,mask=mask))
        self.last_tracking_detections=rows
        return [d for d in rows if d['confidence']>=self.score]

    def metadata(self):return dict(self.info)
