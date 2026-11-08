"""Visual K-way adapters for the prepared PUI comparison environments."""
import hashlib
import contextlib
import io
import json
from pathlib import Path
import subprocess
import time
from contextvars import ContextVar

import cv2
import numpy as np
from PIL import Image

ROOT=Path('/mnt/d/projects/triffid-hua-perception')
_SUPPORT_OVERRIDE = ContextVar('pui_support_override', default=None)


def digest(path):
    sha=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):sha.update(chunk)
    return sha.hexdigest()


def revision(path):
    return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()


def support_inputs():
    from triffid_ugv_perception.supports import SupportBank
    from triffid_ugv_perception.ontology import classes_for
    override = _SUPPORT_OVERRIDE.get()
    report=json.loads((ROOT/'ugv_supports/PUI/import_report.json').read_text())
    names = override[1] if override else report['one_shot_classes']
    classes=tuple(classes_for(','.join(names)))
    selection=SupportBank(override[0] if override else ROOT/'ugv_supports/PUI').select_available(classes)
    return classes,selection


class VisualAdapter:
    def __init__(self,name,repo,checkpoint,size,source_info=None):
        import torch
        self.torch=torch;self.name=name;self.classes,self.selection=support_inputs()
        self.size=size;self.score=.35;self.floor=.1
        self.info=dict(name=name,source_revision=revision(repo) if repo else None,checkpoint_sha256=digest(checkpoint),
            image_size=size,shots_per_class={c.name:len(self.selection.for_class(c)) for c in self.classes},
            bank_sha256=self.selection.manifest_sha256,prompt_mode='visual',query_annotations_used=False)
        self.info.update(source_info or {})
        cache_root=ROOT/'ugv_models'/f'{name.lower().replace("-","_")}_support_cache'/'PUI'
        cache_root.mkdir(parents=True,exist_ok=True)
        key=hashlib.sha256(json.dumps(dict(self.info,adapter_schema=2 if name=='FSOD-VFM' else 1),sort_keys=True).encode()).hexdigest()
        self.cache=cache_root/(key+'.pth')
        self.info['support_cache']=str(self.cache)
        self.info['support_cache_hit']=self.cache.exists()

    def metadata(self):return dict(self.info)

    def row(self,box,score,index,mask=None):
        item=self.classes[int(index)]
        return dict(bbox=list(map(float,box)),confidence=float(np.clip(score,0,1)),
                    class_id=item.id,class_name=item.name,**({'mask':mask} if mask is not None else {}))

    def finish(self,rows):
        self.last_tracking_detections=[r for r in rows if r['confidence']>=self.floor]
        return [r for r in self.last_tracking_detections if r['confidence']>=self.score]


class YoloeAdapter(VisualAdapter):
    def __init__(self):
        path=ROOT/'ugv_models/weights/YOLOE/yoloe-v8l-seg.pt'
        super().__init__('YOLOE',ROOT/'ugv_models/yoloe',path,640)
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
        torch=self.torch
        self.model=YOLOE(str(path));self.model.to('cuda')
        if self.cache.exists():memory=torch.load(self.cache,map_location='cuda',weights_only=False)
        else:
            vectors=[]
            with torch.inference_mode():
                for item in self.classes:
                    examples=[]
                    for support in self.selection.for_class(item):
                        rgb,mask=support.load();y,x=np.where(mask)
                        box=np.array([[x.min(),y.min(),x.max()+1,y.max()+1]],dtype=np.float32)
                        self.model.predict(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),imgsz=self.size,device=0,
                            prompts=dict(bboxes=box,cls=np.array([0])),predictor=YOLOEVPSegPredictor,
                            return_vpe=True,verbose=False,save=False,conf=.99)
                        examples.append(self.model.predictor.vpe[0,0].detach())
                    vectors.append(torch.nn.functional.normalize(torch.stack(examples).mean(0),dim=0))
                    print(f'YOLOE supports: {item.name}: {len(examples)}',flush=True)
            memory=torch.stack(vectors)[None].cpu();torch.save(memory,self.cache)
        if memory.shape[1]!=len(self.classes):raise ValueError('YOLOE cached class count mismatch')
        self.model.set_classes([c.name for c in self.classes],memory.cuda())
        self.model.predictor=None
        self.info.update(backbone='YOLOE-v8-L',support_aggregation='normalized class mean of all selected visual box embeddings',
                         prompt_box_policy='tight union of annotated class mask',precision='float32')

    def detect(self,bgr):
        result=self.model.predict(bgr,imgsz=self.size,device=0,conf=self.floor,iou=.5,
                                  verbose=False,save=False,retina_masks=True)[0]
        rows=[]
        for i,(box,score,label) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy())):
            mask=result.masks.data[i].cpu().numpy()>0.5 if result.masks is not None else None
            rows.append(self.row(box,score,label,mask))
        return self.finish(rows)


class Yoloe26Adapter(VisualAdapter):
    """Released YOLOE-26 with cached K-way visual prototypes; no text encoder."""
    def __init__(self,scale):
        if scale not in ('m','l'):raise ValueError('YOLOE-26 scale must be m or l')
        import ultralytics
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe.predict import YOLOEVPSegPredictor
        name=f'YOLOE-26-{scale.upper()}'
        path=ROOT/f'ugv_models/weights/YOLOE/yoloe-26{scale}-seg.pt'
        super().__init__(name,None,path,640,source_info=dict(
            ultralytics_version=ultralytics.__version__,source_url='https://github.com/ultralytics/ultralytics',
            visual_adapter_schema=2,nms=False))
        torch=self.torch
        self.model=YOLOE(str(path));self.model.to('cuda')
        self.model.model.end2end=True
        if not self.model.model.end2end:raise ValueError('YOLOE-26 checkpoint has no end-to-end head')
        if self.cache.exists():memory=torch.load(self.cache,map_location='cuda',weights_only=True)
        else:
            predictor=YOLOEVPSegPredictor(overrides=dict(task='segment',imgsz=self.size,
                device=0,half=False,nms=False,verbose=False,save=False))
            predictor.setup_model(model=self.model.model,verbose=False)
            if not self.model.model.end2end:raise ValueError('Visual prompt extraction disabled the end-to-end head')
            vectors=[]
            with torch.inference_mode():
                for item in self.classes:
                    examples=[]
                    for support in self.selection.for_class(item):
                        rgb,mask=support.load();y,x=np.where(mask)
                        box=np.array([[x.min(),y.min(),x.max()+1,y.max()+1]],dtype=np.float32)
                        predictor.set_prompts(dict(bboxes=box,cls=np.array([0])))
                        embedding=predictor.get_vpe(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                        if embedding.ndim!=3 or embedding.shape[:2]!=(1,1):
                            raise ValueError('Unexpected YOLOE-26 visual embedding shape')
                        examples.append(embedding[0,0].detach())
                    vectors.append(torch.nn.functional.normalize(torch.stack(examples).mean(0),dim=0))
                    print(f'{name} supports: {item.name}: {len(examples)}',flush=True)
            memory=torch.stack(vectors)[None].cpu();torch.save(memory,self.cache)
        if memory.ndim!=3 or memory.shape[:2]!=(1,len(self.classes)) or not torch.isfinite(memory).all():
            raise ValueError('Invalid YOLOE-26 support cache')
        self.model.set_classes([c.name for c in self.classes],memory.cuda())
        self.model.predictor=None
        self.info.update(backbone=name,architecture='YOLO26 convolutional backbone',
            support_aggregation='normalized class mean of all selected visual box embeddings',
            prompt_box_policy='tight union of annotated class mask',precision='float32',
            end_to_end=bool(self.model.model.end2end),postprocessing='native end-to-end head; no NMS')

    def detect(self,bgr):
        result=self.model.predict(bgr,imgsz=self.size,device=0,conf=self.floor,iou=.5,
            nms=False,agnostic_nms=False,verbose=False,save=False,retina_masks=True)[0]
        if not self.model.predictor.model.end2end:raise ValueError('YOLOE-26 prediction silently selected the NMS branch')
        rows=[]
        for i,(box,score,label) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy())):
            mask=result.masks.data[i].cpu().numpy()>0.5 if result.masks is not None else None
            rows.append(self.row(box,score,label,mask))
        return self.finish(rows)


class DinovAdapter(VisualAdapter):
    def __init__(self):
        repo=ROOT/'ugv_models/dinov';path=ROOT/'ugv_models/weights/DINOv/model_swinT.pth'
        super().__init__('DINOv',repo,path,640)
        from dinov import build_model
        from utils.arguments import load_opt_from_config_file
        torch=self.torch
        opt=load_opt_from_config_file(str(repo/'configs/dinov_sam_coco_train.yaml'))
        opt['MODEL']['BACKBONE']['LOAD_PRETRAINED']=False
        opt['OUTPUT_DIR']=str(self.cache.parent)
        with contextlib.redirect_stdout(io.StringIO()):
            self.model=build_model(opt)
        state=torch.load(path,map_location='cpu',weights_only=False);state=state.get('model',state)
        state={k[6:]:v for k,v in state.items() if k.startswith('model.')}
        if 'criterion.empty_weight' not in self.model.state_dict():state.pop('criterion.empty_weight',None)
        self.model.load_state_dict(state,strict=True);del state
        self.model.cuda().eval().requires_grad_(False)
        if self.cache.exists():memory=torch.load(self.cache,map_location='cuda',weights_only=False)
        else:
            content=[];boxes=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                for item in self.classes:
                    examples=[]
                    for support in self.selection.for_class(item):
                        rgb,mask=support.load();data=self.input(rgb)
                        h,w=data['image'].shape[-2:]
                        shape=torch.from_numpy(cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)[None]).cuda()
                        scales,_,ph,pw=self.model.get_encoder_feature([data])
                        tokens,bbox,_=self.model.get_visual_prompt_content_feature(scales,shape,ph,pw)
                        examples.append(tokens.float())
                    content.append(torch.stack(examples).mean(0));boxes.append(bbox.float())
                    print(f'DINOv supports: {item.name}: {len(examples)}',flush=True)
            memory=dict(content=torch.cat(content,dim=1).cpu(),boxes=torch.cat(boxes,dim=1).cpu())
            torch.save(memory,self.cache)
        self.content=memory['content'].cuda();self.boxes=memory['boxes'].cuda()
        count=self.content.shape[1];per_class=count//len(self.classes)
        if count%len(self.classes):raise ValueError('DINOv content/class count mismatch')
        self.attention=torch.ones(count,count,device='cuda',dtype=torch.bool)
        for i in range(len(self.classes)):
            self.attention[i*per_class:(i+1)*per_class,i*per_class:(i+1)*per_class]=False
        self.info.update(backbone='Swin-T',joint_class_count=len(self.classes),precision='float16 autocast',
            support_aggregation='per-class mean of native visual content tokens; class-isolated content attention',
            preprocessing='RGB; short side 640; nearest support masks',nms_iou=.5)

    def input(self,rgb):
        h,w=rgb.shape[:2];scale=self.size/min(h,w)
        resized=np.asarray(Image.fromarray(rgb).resize((round(w*scale),round(h*scale)),Image.Resampling.BICUBIC))
        return dict(image=self.torch.from_numpy(resized.copy()).permute(2,0,1).cuda(),height=h,width=w)

    def detect(self,bgr):
        torch=self.torch;h,w=bgr.shape[:2]
        from torchvision.ops import batched_nms
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            data=self.input(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            scales,masks,ph,pw=self.model.get_encoder_feature([data])
            output,_=self.model.sem_seg_head.predictor.forward_openset_image_with_extracted_content(
                scales,masks,None,self.content,self.boxes,self.attention,None,
                extra=dict(part=False,whole=False,seg=True,det=True))
            logits=output['pred_logits'][0].float()
            if logits.shape[-1]!=len(self.classes):raise ValueError(f'DINOv logits shape {logits.shape} does not match K-way prompts')
            scores,labels=logits.sigmoid().max(-1)
            chosen=torch.where(scores>=self.floor)[0]
            # Native pred_boxes are normalized cxcywh within the padded encoder image.
            resized_h,resized_w=data['image'].shape[-2:]
            boxes=output['pred_boxes'][0][chosen].float()
            xyxy=torch.cat((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),dim=1)*boxes.new_tensor([pw/resized_w*w,ph/resized_h*h,pw/resized_w*w,ph/resized_h*h])
            xyxy[:,0::2]=xyxy[:,0::2].clamp(0,w);xyxy[:,1::2]=xyxy[:,1::2].clamp(0,h)
            keep=batched_nms(xyxy,scores[chosen],labels[chosen],.5)[:100]
            rows=[]
            # Upsample selected masks individually to bound peak VRAM at native RGB resolution.
            resized_h,resized_w=data['image'].shape[-2:]
            for index in keep.tolist():
                q=chosen[index]
                mask=output['pred_masks'][0,q][None,None].float()
                mask=torch.nn.functional.interpolate(mask,size=(ph,pw),mode='bilinear',align_corners=False)
                mask=mask[:,:,:resized_h,:resized_w]
                mask=torch.nn.functional.interpolate(mask,size=(h,w),mode='bilinear',align_corners=False)[0,0]>0
                box=xyxy[index].cpu().tolist()
                if box[2]>box[0] and box[3]>box[1]:rows.append(self.row(box,scores[q].item(),labels[q].item(),mask.cpu().numpy()))
        return self.finish(rows)


def area_mask_weights(mask,shape):
    # A thin annotated object can vanish under binary point sampling at a 45x45
    # feature grid. Area coverage retains its real foreground contribution.
    height,width=shape
    return cv2.resize(mask.astype(np.float32),(int(width),int(height)),interpolation=cv2.INTER_AREA)


class FsodAdapter(VisualAdapter):
    def __init__(self):
        repo=ROOT/'ugv_models/fsod_vfm';checkpoint=repo/'checkpoints/dinov2_vitl14_pretrain.pth'
        super().__init__('FSOD-VFM',repo,checkpoint,630)
        torch=self.torch
        from model.dinov2 import get_dinov2_transform
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from chatrex.upn import UPNWrapper
        self.features=torch.hub.load(str(repo/'dinov2'),'dinov2_vitl14',source='local',pretrained=False)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.features.load_state_dict(state,strict=True);del state
        self.features.cuda().eval().requires_grad_(False)
        self.transform=get_dinov2_transform((630,630))
        self.sam=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',
            str(ROOT/'ugv_models/weights/FSOD-VFM/sam2.1_hiera_large.pt'),device='cuda'))
        self.upn=UPNWrapper(str(ROOT/'ugv_models/weights/FSOD-VFM/upn_large.pth'))
        if self.cache.exists():self.prototypes=torch.load(self.cache,map_location='cuda',weights_only=False)
        else:
            from support_util import resize_mask_to_features
            grouped={}
            for i,item in enumerate(self.classes):
                for support in self.selection.for_class(item):grouped.setdefault(support.image,[]).append((i,support))
            vectors=[[] for _ in self.classes]
            for image_path,examples in grouped.items():
                pil=Image.open(image_path).convert('RGB');features=self.encode(pil)
                for i,support in examples:
                    _,mask=support.load();small=area_mask_weights(mask,features.shape[-2:])
                    weight=torch.from_numpy(small).cuda()
                    if not weight.any():raise ValueError(f'Support vanished at feature resolution: {support.example_id}')
                    vectors[i].append((features*weight).sum((2,3))[0]/weight.sum())
                print(f'FSOD-VFM support image: {image_path.name}',flush=True)
            self.prototypes=torch.nn.functional.normalize(torch.stack([torch.stack(v).mean(0) for v in vectors],dim=1),dim=0)
            torch.save(self.prototypes.cpu(),self.cache)
        self.info.update(backbone='DINOv2 ViT-L/14',proposal='UPN-L coarse_grained_prompt',segmenter='SAM2.1-L',
            proposal_checkpoint_sha256=digest(ROOT/'ugv_models/weights/FSOD-VFM/upn_large.pth'),
            mask_checkpoint_sha256=digest(ROOT/'ugv_models/weights/FSOD-VFM/sam2.1_hiera_large.pt'),
            proposal_min_score=.01,max_proposals=100,sam_batch_size=8,diffusion_steps=30,alpha=.3,decay_lambda=.5,
            support_mask_policy='Human annotated class masks instead of SAM-generated support masks; area-weighted pooling retains tiny supports',
            support_aggregation='native mean support prototype and L2 normalization',precision='float32')

    def encode(self,pil):
        from support_util import get_dinov2_features
        with self.torch.inference_mode():return get_dinov2_features(self.features,self.transform,pil,device='cuda')

    def detect(self,bgr):
        torch=self.torch
        from support_util import resize_mask_to_features
        from metric import graph_diffusion_ios
        pil=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));h,w=bgr.shape[:2]
        with torch.inference_mode():
            proposals=self.upn.filter(self.upn.inference(pil,'coarse_grained_prompt'),min_score=.01,nms_value=1)
            if not proposals['original_xyxy_boxes']:return self.finish([])
            boxes=proposals['original_xyxy_boxes'][0];scores=proposals['scores'][0]
            order=np.argsort(-np.asarray(scores))[:100]
            boxes=np.asarray(boxes)[order].reshape(-1,4).copy()
            boxes[:,0::2]=boxes[:,0::2].clip(0,w);boxes[:,1::2]=boxes[:,1::2].clip(0,h)
            boxes=boxes[(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])]
            if not len(boxes):return self.finish([])
            features=self.encode(pil);self.sam.set_image(pil)
            rows=[];small_masks=[];labels=[]
            for begin in range(0,len(boxes),8):
                masks,_,_=self.sam.predict(point_coords=None,point_labels=None,box=boxes[begin:begin+8],multimask_output=False)
                if masks.ndim==4:masks=masks[:,0]
                for box,mask in zip(boxes[begin:begin+8],masks):
                    weight=torch.from_numpy(resize_mask_to_features(mask,features.shape[-2:])).cuda()
                    vector=torch.nn.functional.normalize((features*weight).sum((2,3))/(weight.sum()+1e-7),dim=1,eps=1e-2)
                    similarities=vector@self.prototypes;score,label=similarities.max(1)
                    rows.append(self.row(box,score.item(),label.item(),mask.astype(bool)))
                    small_masks.append(weight[None,None]);labels.append(label.item())
            if rows:
                compact=torch.stack(small_masks);label_tensor=torch.tensor(labels,device='cuda')
                ios=graph_diffusion_ios(compact,label_tensor,len(self.classes),max_iter=30,alpha=.3)
                for i,row in enumerate(rows):row['confidence']=float(max(0.,row['confidence']*max(0.,1-ios[i].item())**.5))
        return self.finish(rows)


def build_detector(name, support_bank=None, class_names=None):
    if (support_bank is None) != (class_names is None):
        raise ValueError('Supply both evaluation support bank and classes')
    token = _SUPPORT_OVERRIDE.set((support_bank,class_names) if support_bank else None)
    try:
        return _build_detector(name)
    finally:
        _SUPPORT_OVERRIDE.reset(token)


def _build_detector(name):
    if name=='YOLOE':return YoloeAdapter()
    if name in ('YOLOE-26-M','YOLOE-26-L'):return Yoloe26Adapter(name[-1].lower())
    if name=='DINOv':return DinovAdapter()
    if name=='FSOD-VFM':return FsodAdapter()
    from triffid_ugv_perception.config import NODE_PARAMETERS
    from triffid_ugv_perception.models import create_detector
    report=json.loads((ROOT/'ugv_supports/PUI/import_report.json').read_text())
    override = _SUPPORT_OVERRIDE.get()
    cfg=dict(NODE_PARAMETERS,device='cuda',classes=','.join(override[1] if override else report['one_shot_classes']),shots=1,
        support_bank=str(override[0] if override else ROOT/'ugv_supports/PUI'),confidence_threshold=.35,
        pet_tracking_floor=.1,pet_prompt_mode='visual',pet_use_all_supports=True,
        pet_repo=str(ROOT/'ugv_models/pet_dino'),pet_checkpoint=str(ROOT/'ugv_models/pet_dino_swin_t.pth'),
        pet_bert=str(ROOT/'ugv_models/pet_bert'),pet_support_cache=str(ROOT/'ugv_models/pet_support_cache/PUI'),
        pet_precision='float16',pet_max_size=1333,pet_backbone='swin-t',
        opus_repo=str(ROOT/'ugv_models/opus'),opus_checkpoint=str(ROOT/'ugv_models/opus_dinov3_convnext_b.pth'),
        opus_support_cache=str(ROOT/'ugv_models/opus_support_cache/PUI'),opus_precision='float16',
        opus_use_all_supports=True,opus_prompt_mode='visual')
    if name in ('SINE','COSINE','DE-ViT','SANSA'):
        cfg.update(backend={'SINE':'sine','COSINE':'cosine','DE-ViT':'devit','SANSA':'sansa'}[name],image_size=640,
            sine_support_cache=str(ROOT/'ugv_models/sine_support_cache/ROSBAG-PUI'),
            cosine_repo=str(ROOT/'ugv_models/cosine'),cosine_checkpoint=str(ROOT/'ugv_models/cosine_multiscale.bin'),
            cosine_support_cache=str(ROOT/'ugv_models/cosine_support_cache/ROSBAG-PUI'),
            devit_repo=str(ROOT/'ugv_models/devit'),devit_support_cache=str(ROOT/'ugv_models/devit_support_cache/ROSBAG-PUI'),
            devit_checkpoint=str(ROOT/'ugv_models/devit/weights/vits_0089999.pth'),
            devit_background=str(ROOT/'ugv_models/devit/weights/background_prototypes.vits14.pth'),
            sansa_repo='/mnt/d/Models/ThirdParty/SANSA',checkpoint='/mnt/d/Models/Checkpoints/SANSA/adapter_generalist.pth',
            sam2_checkpoint='/mnt/d/Models/Checkpoints/SANSA/sam2_hiera_large.pt')
        if name in ('SINE','COSINE'):
            cfg['confidence_threshold']=.7
        if name=='DE-ViT':
            cfg['confidence_threshold']=.45
    elif name in ('PET-DINO','OPUS'):
        cfg.update(backend='pet_dino' if name=='PET-DINO' else 'opus',image_size=800 if name=='PET-DINO' else 640)
    else:
        raise ValueError(f'Unknown model: {name}')
    return create_detector(cfg)
