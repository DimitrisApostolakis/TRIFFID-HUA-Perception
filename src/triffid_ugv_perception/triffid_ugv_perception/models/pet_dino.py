"""Official PET-DINO with Visual-G support embeddings or class-name text prompts."""
import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
from ..ontology import classes_for
from ..supports import SupportBank
from .devit import file_hash


def support_box(mask):
    """A support prompt must surround one instance, not disconnected objects."""
    count,_,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    if count<2:raise ValueError('Support mask has no foreground')
    x,y,w,h,_=stats[1+np.argmax(stats[1:,cv2.CC_STAT_AREA])]
    return [int(x),int(y),int(x+w),int(y+h)]


class PetDinoDetector:
    def __init__(self,config):
        import torch
        self.torch=torch;self.device=config['device'];self.score=float(config['confidence_threshold'])
        self.tracking_score=float(config.get('pet_tracking_floor',self.score))
        self.prompt_mode=config.get('pet_prompt_mode','visual')
        if self.prompt_mode not in ('visual','text'):raise ValueError('Unsupported PET prompt mode')
        self.classes=tuple(classes_for(config['classes']))
        self.selection=None
        if self.prompt_mode=='visual':
            bank=SupportBank(config['support_bank'])
            self.selection=bank.select_available(self.classes) if config.get('pet_use_all_supports',False) else bank.select(self.classes,config['shots'])
        self.repo=Path(config['pet_repo']).resolve();sys.path.insert(0,str(self.repo))
        self.size=int(config['image_size']);self.max_size=int(config['pet_max_size'])
        self.query_encodings=0;self.half=config['pet_precision']=='float16' and self.device.startswith('cuda')
        from mmengine.config import Config
        from mmdet.utils import register_all_modules
        register_all_modules()
        from mmdet.registry import MODELS
        from transformers import BertModel
        self.backbone=config.get('pet_backbone','swin-t')
        if self.backbone not in ('swin-t','swin-l'):raise ValueError('Unsupported PET-DINO backbone')
        cfg=Config.fromfile(str(self.repo/f'configs/pet_dino/pet_dino_{self.backbone}_8xb4_12e_obj365.py'))
        cfg.model.language_model.name=config['pet_bert']
        cfg.model.backbone.init_cfg=None
        cfg.model.memory_bank={}
        cfg.model.backbone.with_cp=False
        cfg.model.encoder.num_cp=0
        def disable_checkpointing(value):
            if isinstance(value,dict):
                if 'num_cp' in value:value['num_cp']=0
                for child in value.values():disable_checkpointing(child)
            elif isinstance(value,(tuple,list)):
                for child in value:disable_checkpointing(child)
        disable_checkpointing(cfg.model)
        original=BertModel.from_pretrained
        # Full PET checkpoint includes BERT; construct from configuration only.
        BertModel.from_pretrained=classmethod(lambda cls,*args,**kwargs:cls(kwargs['config'],add_pooling_layer=kwargs.get('add_pooling_layer',False)))
        try:self.model=MODELS.build(cfg.model)
        finally:BertModel.from_pretrained=original
        checkpoint=Path(config['pet_checkpoint'])
        self.checkpoint_hash=file_hash(checkpoint)
        state=torch.load(checkpoint,map_location='cpu');state=state.get('state_dict',state)
        self.model.load_state_dict(state,strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.revision=subprocess.check_output(['git','-C',str(self.repo),'rev-parse','HEAD'],text=True).strip()
        self.model.backbone.register_forward_hook(self._features)
        self.preparing=True
        if self.prompt_mode=='text':
            self.cache=None;self.memory=None
            self.preparing=False;self.last_features=None
            return
        identity=dict(supports=self.selection.metadata(),checkpoint=self.checkpoint_hash,backbone=self.backbone,source_revision=self.revision,size=self.size,max_size=self.max_size,precision=config['pet_precision'],prompt_box_policy='largest_connected_component')
        directory=Path(config['pet_support_cache']);directory.mkdir(parents=True,exist_ok=True)
        self.cache=directory/(hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()+'.pth')
        if self.cache.exists():self.memory=torch.load(self.cache,map_location=self.device)
        else:
            vectors=[]
            with torch.inference_mode(),self._amp():
                for index,item in enumerate(self.classes):
                    examples=[]
                    for support in self.selection.for_class(item):
                        rgb,mask=support.load();box=support_box(mask)
                        data=self._input(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),box,index)
                        vector=self.model.extract_visual_embedding(data)[0]
                        examples.append(vector[0])
                    stacked=torch.stack(examples)
                    if hasattr(self.model,'visual_prompt_enhancer') and len(stacked)>1:
                        attention=torch.ones((1,len(stacked),len(stacked)),device=self.device,dtype=torch.bool)
                        stacked=self.model.visual_prompt_enhancer(memory_visual=stacked[None],visual_self_attention_masks=attention)[0]
                    vectors.append(stacked.mean(0))
            self.memory=torch.stack(vectors).detach()
            torch.save(self.memory.cpu(),self.cache)
        self.preparing=False;self.last_features=None

    def _amp(self):
        return self.torch.cuda.amp.autocast(dtype=self.torch.float16) if self.half else contextlib.nullcontext()

    def _features(self,module,inputs,output):
        if not self.preparing:self.query_encodings+=1
        self.last_features=output[0].detach()

    def _input(self,bgr,box=None,index=None):
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData
        torch=self.torch;h,w=bgr.shape[:2]
        scale=min(self.size/min(h,w),self.max_size/max(h,w));nw,nh=round(w*scale),round(h*scale)
        resized=cv2.resize(bgr,(nw,nh));sx,sy=nw/w,nh/h
        sample=DetDataSample(metainfo=dict(img_id=0,text=[c.name for c in self.classes],custom_entities=True,ori_shape=(h,w),img_shape=(nh,nw),scale_factor=(sx,sy),
            prompt_type='Visual' if self.prompt_mode=='visual' else 'Text',prompt_ori_shape=(h,w),prompt_img_shape=(nh,nw),
            homography_matrix=np.diag([sx,sy,1.]).astype(np.float32),prompt_bboxes_already_transformed=True))
        sample.gt_instances=InstanceData(bboxes=torch.empty((0,4)),labels=torch.empty((0,),dtype=torch.long))
        if box is not None:
            sample.set_metainfo(dict(prompt_bboxes=torch.tensor([box],dtype=torch.float32)*torch.tensor([sx,sy,sx,sy]),prompt_bboxes_labels=torch.tensor([index])))
        elif self.prompt_mode=='visual':
            sample.set_metainfo(dict(memory_visual=self.memory,memory_label=torch.arange(len(self.classes)),memory_class_name=[c.name for c in self.classes]))
        return dict(inputs=[torch.from_numpy(resized.transpose(2,0,1).copy())],data_samples=[sample])

    def detect(self,bgr):
        torch=self.torch
        with torch.inference_mode(),self._amp():
            data=self.model.data_preprocessor(self._input(bgr),False)
            output=self.model.predict(data['inputs'],data['data_samples'])[0].pred_instances
        result=[];self.last_candidates=[];self.last_tracking_detections=[]
        for box,score,label in zip(output.bboxes.detach().cpu().numpy(),output.scores.detach().cpu().numpy(),output.labels.detach().cpu().numpy()):
            item=self.classes[int(label)]
            x1,y1,x2,y2=box;h,w=bgr.shape[:2]
            box=np.clip(box,[0,0,0,0],[w,h,w,h])
            if box[2]<=box[0] or box[3]<=box[1]:continue
            self.last_candidates.append(dict(bbox=box.tolist(),confidence=float(score),class_id=item.id,class_name=item.name))
            if float(score)<min(self.score,self.tracking_score):continue
            feat=self.last_features[0].float();fh,fw=feat.shape[-2:]
            a,b=max(0,int(x1/w*fw)),max(0,int(y1/h*fh));c,d=min(fw,max(a+1,int(np.ceil(x2/w*fw)))),min(fh,max(b+1,int(np.ceil(y2/h*fh))))
            vector=torch.nn.functional.normalize(feat[:,b:d,a:c].mean((1,2)),dim=0).cpu().numpy().tolist()
            detection=dict(bbox=box.tolist(),confidence=float(score),class_id=item.id,class_name=item.name,appearance=vector)
            self.last_tracking_detections.append(detection)
            if float(score)>=self.score:result.append(detection)
        self.last_features=None
        return result

    def metadata(self):
        return dict(name='pet_dino',checkpoint_sha256=self.checkpoint_hash,source_revision=self.revision,
            backbone=self.backbone,prompt_mode=self.prompt_mode,text_prompts=[item.name for item in self.classes],
            support_bank=self.selection.metadata() if self.selection else None,
            support_cache=str(self.cache) if self.cache else None,query_encodings=self.query_encodings,
            shots_per_class={item.name:len(self.selection.for_class(item)) if self.selection else 0 for item in self.classes},
            prompt_box_policy='largest_connected_component' if self.selection else None,
            mode='Visual-G; class means from annotated support bounding boxes; no query annotations' if self.selection else 'Text; exact ontology class names; no visual support embeddings',strict_checkpoint_load=True,
            image_size=self.size,max_size=self.max_size)
