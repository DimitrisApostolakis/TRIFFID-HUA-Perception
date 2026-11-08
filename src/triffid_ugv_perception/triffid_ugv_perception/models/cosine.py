"""Released multi-scale COSINE components, cached K-way visual/text prompts.

The public FSOD package references absent model variants. This adapter uses
the released shared encoder, pixel decoder and MFormer directly, with strict
checkpoint loading, and the published FC-CLIP scoring equations. No training.
"""
import hashlib
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace, MethodType

import cv2
import numpy as np
from ..ontology import classes_for
from ..supports import SupportBank
from .devit import file_hash
from .sine import SineDetector


def single_clip_forward(encoder, pixels):
    """Public wrapper repeats the same frozen backbone forward for every level."""
    features=encoder.encoder(pixels)
    return [features[level] for level in encoder.feature_layer_list]


def native_clip_text(encoder,text,normalize=False):
    """Delegate token layout to the installed OpenCLIP version itself."""
    return encoder.clip_model.encode_text(text,normalize=normalize)


def install_msda_bridge():
    """Use MMCV's precompiled version of the same deformable-attention op."""
    if 'MultiScaleDeformableAttention' in sys.modules:return
    try:
        __import__('MultiScaleDeformableAttention')
        return
    except ModuleNotFoundError:pass
    import mmcv._ext as extension
    bridge=ModuleType('MultiScaleDeformableAttention')
    def forward(value,shapes,starts,locations,weights,step):
        return extension.ms_deform_attn_forward(value,shapes,starts,locations,weights,im2col_step=step)
    def backward(*args):raise RuntimeError('COSINE inference bridge does not support training')
    bridge.ms_deform_attn_forward=forward;bridge.ms_deform_attn_backward=backward
    sys.modules[bridge.__name__]=bridge


def build_components(repo):
    """Construct published architecture without redundant pretrained downloads.

    Every encoder tensor must subsequently load from the full COSINE checkpoint.
    Compatibility hooks apply only during construction and are restored.
    """
    import torch
    import open_clip
    from detectron2.layers import batch_norm
    sys.path.insert(0,str(repo))
    install_msda_bridge()
    import dinov2.utils.utils as dino_utils
    if Path(dino_utils.__file__).resolve()!=repo/'dinov2/utils/utils.py':
        raise RuntimeError('Another DINOv2 checkout shadows COSINE; initialize this backend in a fresh process')
    from cosine.model.image_encoder import dino_ms, clip_ms_fsod
    from cosine.model.image_encoder.encodersoup_ms_fsod import EncoderSoup
    from cosine.model.pixel_decoder import msdeformattn
    from cosine.model.transformer_decoder import mformer_ms
    args=SimpleNamespace(dinov2_size='vit_large',dinov2_weights='',clip_weights=None,feat_chans=256,
        neck_in_features='p2||p3||p4||p5',neck_encoder_in_features='p3||p4||p5',
        neck_conv_dim=256,neck_mask_dim=256,neck_transformer_dropout=0.,neck_transformer_nheads=8,
        neck_dim_feedforward=1024,neck_encoder_layers=6,neck_common_stride=4,
        transformer_depth=6,transformer_nheads=8,transformer_mlp_dim=2048,transformer_mask_dim=256,
        transformer_fusion_layer_depth=1,transformer_num_queries=200,transformer_pre_norm=True)
    original_create=open_clip.create_model_and_transforms
    original_load=dino_utils.load_pretrained_weights
    modules=(dino_ms,clip_ms_fsod,msdeformattn,mformer_ms)
    originals=[module.get_norm for module in modules]
    def uninitialized_clip(*args,**kwargs):
        kwargs['pretrained']=None
        return original_create(*args,**kwargs)
    def compatible_norm(name,channels):
        return dino_ms.LayerNorm2d(channels) if name=='LN' else batch_norm.get_norm(name,channels)
    try:
        open_clip.create_model_and_transforms=uninitialized_clip
        dino_utils.load_pretrained_weights=lambda *args,**kwargs:None
        for module in modules:module.get_norm=compatible_norm
        model=torch.nn.Module()
        model.encoder_soup=EncoderSoup(dino_ms.build_dinov2(args),clip_ms_fsod.build_clip(args))
        model.pixel_decoder=msdeformattn.build_pixel_decoder(args,model.encoder_soup.clip.output_shape())
        model.transformer_decoder=mformer_ms.build_mformer(args)
    finally:
        open_clip.create_model_and_transforms=original_create
        dino_utils.load_pretrained_weights=original_load
        for module,original in zip(modules,originals):module.get_norm=original
    # Compute each CLIP pyramid from one forward rather than five identical
    # forwards. This changes no feature values in frozen evaluation mode.
    model.encoder_soup.clip.get_enc_embs=MethodType(single_clip_forward,model.encoder_soup.clip)
    # COSINE's copied encode_text assumes batch-first OpenCLIP. The installed
    # 2.24 transformer is sequence-first; its own API handles the permutation.
    model.encoder_soup.clip.encoder.encode_text=MethodType(native_clip_text,model.encoder_soup.clip.encoder)
    return model


class CosineDetector(SineDetector):
    def __init__(self,config):
        import torch
        self.torch=torch
        self.classes=tuple(sorted(classes_for(config['classes']),key=lambda c:c.id))
        self.by_id={c.id:c for c in self.classes}
        self.selection=SupportBank(config['support_bank']).select(self.classes,config['shots'])
        self.repo=Path(config['cosine_repo']).resolve()
        checkpoint=Path(config['cosine_checkpoint']).resolve()
        if not checkpoint.is_file():raise ValueError(f'COSINE checkpoint missing: {checkpoint}')
        self.checkpoint_hash=file_hash(checkpoint)
        expected=config.get('cosine_checkpoint_sha256','8f85b892627eacdf0e63e0c1a147af5da681611ba9c32c9f6cf08af755d16fac')
        if self.checkpoint_hash!=expected:
            raise ValueError('COSINE multi-scale checkpoint checksum mismatch or download incomplete')
        self.device=str(config['device'])
        if self.device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
        self.precision=config['cosine_precision'];self.half=self.precision=='float16' and self.device.startswith('cuda')
        if self.precision not in ('float16','float32'):raise ValueError('Invalid COSINE precision')
        # Neck features must align: DINO patches at 14px and CLIP at 16px.
        self.size=int(np.ceil(int(config['image_size'])/28)*28)
        self.clip_size=self.size//14*16
        self.score=float(config['confidence_threshold']);self.nms_iou=float(config['cosine_mask_nms_iou'])
        if self.size<56 or not 0<=self.score<=1 or not 0<self.nms_iou<=1:raise ValueError('Invalid COSINE inference settings')
        self.model=build_components(self.repo)
        state=torch.load(checkpoint,map_location='cpu');state=state.get('model',state)
        # The released encoder adds `neck` as an alias of simfp_4 for support
        # pooling. The checkpoint saves only simfp_4. Restore duplicate names
        # from their exact shared tensors, then keep strict loading enabled.
        if self.model.encoder_soup.dinov2.neck is not self.model.encoder_soup.dinov2.simfp_4:
            raise ValueError('Unexpected COSINE support-neck architecture')
        self.restored_aliases=[]
        for name in self.model.encoder_soup.dinov2.neck.state_dict():
            alias='encoder_soup.dinov2.neck.'+name
            canonical='encoder_soup.dinov2.simfp_4.'+name
            if alias not in state:
                if canonical not in state:raise ValueError(f'Checkpoint omits support-neck weight: {canonical}')
                state[alias]=state[canonical]
                self.restored_aliases.append(alias)
        training_keys={key for key in state if key.startswith('criterion.')}
        if training_keys-{'criterion.empty_weight'}:raise ValueError(f'Unexpected training weights: {training_keys}')
        self.model.load_state_dict({key:value for key,value in state.items() if key not in training_keys},strict=True)
        del state
        # The official pixel decoder explicitly computes deformable attention
        # in FP32. Retain FP32 weights and use autocast for the other components.
        self.model.to(self.device).eval().requires_grad_(False)
        from detectron2.data.transforms import ResizeShortestEdge
        self.resize=ResizeShortestEdge(self.size,self.size)
        self.revision=subprocess.check_output(['git','-C',str(self.repo),'rev-parse','HEAD'],text=True).strip()
        self.query_encodings=0;self.clip_query_encodings=0;self._last_features=None
        identity=dict(schema='triffid.cosine.supports.v1',supports=self.selection.metadata(),
            checkpoint_sha256=self.checkpoint_hash,source_revision=self.revision,image_size=self.size,
            clip_image_size=self.clip_size,precision=self.precision,visual_text=True,fcclip=True)
        directory=Path(config['cosine_support_cache']);directory.mkdir(parents=True,exist_ok=True)
        self.cache=directory/(hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()+'.pth')
        if self.cache.is_file():
            cached=torch.load(self.cache,map_location='cpu')
            if cached['metadata']!=identity:raise ValueError('COSINE support cache mismatch')
            self.prompts={key:[value.to(self.device)] for key,value in cached['prompts'].items()}
            self.text_classifier=cached['text_classifier'].to(self.device)
        else:
            self._prepare_supports()
            fd,temporary=tempfile.mkstemp(dir=directory,prefix='.cosine-',suffix='.pth');os.close(fd)
            try:
                torch.save(dict(metadata=identity,prompts={k:v[0].cpu() for k,v in self.prompts.items()},
                    text_classifier=self.text_classifier.cpu()),temporary)
                os.replace(temporary,self.cache)
            finally:
                if os.path.exists(temporary):os.unlink(temporary)
        if self.prompts['sem_prompt_label_list'][0].tolist()!=[c.id for c in self.classes]:
            raise ValueError('COSINE support labels do not match requested ontology IDs')
        from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
        # Read the released category constant without importing dataset
        # registrations, which require unrelated newer LVIS/Detectron2 files.
        tree=ast.parse((self.repo/'cosine/data/load_data/object365.py').read_text())
        O365_CATEGORIES=next(ast.literal_eval(node.value) for node in tree.body
            if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='O365_CATEGORIES' for target in node.targets))
        seen={c['name'].lower() for c in COCO_CATEGORIES}
        seen.update(s.strip().lower() for c in O365_CATEGORIES for s in c['name'].split('/'))
        self.ensemble_power=torch.tensor([.4 if c.name in seen else .8 for c in self.classes],device=self.device)

    def _input(self,bgr):
        torch=self.torch;F=torch.nn.functional
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        transform=self.resize.get_transform(rgb);resized=transform.apply_image(rgb)
        raw=torch.as_tensor(np.ascontiguousarray(resized.transpose(2,0,1))).float()
        dino=raw/255
        dino=(dino-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        padding=(0,self.size-raw.shape[2],0,self.size-raw.shape[1])
        dino=F.pad(dino,padding)
        valid=F.pad(torch.ones_like(raw),padding)
        clip=F.interpolate(F.pad(raw,padding)[None],size=self.clip_size,mode='bilinear',align_corners=False,antialias=True)[0]
        valid=F.interpolate(valid[None],size=self.clip_size,mode='nearest')[0]
        clip=(clip-torch.tensor([122.7709383,116.7460125,104.09373615])[:,None,None])/torch.tensor([68.5005327,66.6321579,70.32316305])[:,None,None]
        return dict(image=dino,clip_image=clip*valid),transform,resized.shape[:2]

    def _prepare_supports(self):
        torch=self.torch;F=torch.nn.functional;soup=self.model.encoder_soup
        from cosine.model.image_encoder.encodersoup_ms_fsod import VILD_PROMPT
        grouped={}
        for item in self.classes:
            for support in self.selection.for_class(item):grouped.setdefault(support.image,[]).append((item,support))
        collected={c.id:[] for c in self.classes}
        with torch.inference_mode(),self._amp():
            text=soup.encode_text_prompt([c.name for c in self.classes]).float()
            text_by_id={c.id:text[index] for index,c in enumerate(self.classes)}
            for path,entries in grouped.items():
                inputs,transform,_=self._input(cv2.imread(str(path)))
                masks=[];labels=[]
                for item,support in entries:
                    _,binary=support.load();mask=transform.apply_segmentation(binary)
                    if not mask.any():raise ValueError('COSINE support foreground vanished during resizing')
                    masks.append(F.pad(torch.from_numpy(mask.copy()),(0,self.size-mask.shape[1],0,self.size-mask.shape[0])))
                    labels.append(item.id)
                features=soup.dinov2.neck(soup.dinov2.get_enc_embs(inputs['image'][None].to(self.device)))
                _,queries,query_labels=soup.encode_vision_prompt(features,torch.stack(masks).to(self.device),torch.tensor(labels,device=self.device))
                for label,query in zip(query_labels.tolist(),queries):
                    # Match FSOD preprocessing: fuse/normalize each image's
                    # visual and text prompt, then average shots per class.
                    collected[label].append(F.normalize((query.float()+text_by_id[label])/2,dim=-1))
            semantic=torch.stack([torch.stack(collected[c.id]).mean(0) for c in self.classes])
            labels=torch.tensor([c.id for c in self.classes],device=self.device)
            self.prompts=dict(id_prompt_list=[semantic.new_empty((0,256))],sem_prompt_list=[semantic],
                id_prompt_label_list=[labels.new_empty((0,))],sem_prompt_label_list=[labels])
            # FC-CLIP text features before COSINE's learned text projection.
            phrases=[template.format(c.name) for c in self.classes for template in VILD_PROMPT]
            chunks=[soup.clip.encoder.get_text_classifier(phrases[i:i+128],self.device).float() for i in range(0,len(phrases),128)]
            text=F.normalize(torch.cat(chunks),dim=-1).reshape(len(self.classes),len(VILD_PROMPT),-1).mean(1)
            self.text_classifier=F.normalize(text,dim=-1)

    def detect(self,bgr):
        torch=self.torch;F=torch.nn.functional
        inputs,_,self._resized_shape=self._input(bgr);soup=self.model.encoder_soup
        try:
            with torch.inference_mode(),self._amp():
                encoded=soup.encode_image(inputs['image'][None].to(self.device),inputs['clip_image'][None].to(self.device))
                self.query_encodings+=1;self.clip_query_encodings+=1
                self._last_features=encoded['dino']['p4'].detach()
                dense=encoded['clip_vis_dense']
                merged=soup.merge_feature(encoded,self.prompts)
                mask_features,_,features=self.model.pixel_decoder.forward_features(merged)
                output=self.model.transformer_decoder(features=features,mask_features=mask_features,prompts=self.prompts,tags=['visual'])[0]
                logits=output['pred_ins_logits'].float();masks=output['pred_ins_masks']
                pooled=soup.mask_pooling(dense,F.interpolate(masks,size=dense.shape[-2:],mode='bilinear',align_corners=False))
                pooled=soup.clip.encoder.visual_prediction_forward(pooled).float()
                external=(soup.clip.encoder.clip_model.logit_scale.float().exp().clamp(max=100)*F.normalize(pooled,dim=-1)@self.text_classifier.T).softmax(-1)
                internal=logits[...,:-1].softmax(-1)
                power=self.ensemble_power
                fused=((1-power)*internal.clamp_min(1e-8).log()+power*external.clamp_min(1e-8).log()).softmax(-1)
                void=logits.softmax(-1)[...,-1:]
                probabilities=fused*(1-void)
                # Official instance scoring: class probability times mean mask
                # confidence, then return class probability for retained masks.
                probabilities,indices=probabilities.flatten().topk(min(100,probabilities.numel()))
                query=indices//len(self.classes);category=indices%len(self.classes)
                # Top-k is independent of mask resizing. Resize only selected
                # hypotheses instead of allocating 200 full-resolution masks.
                resized=F.interpolate(masks[0,query,None].float(),size=(self.size,self.size),mode='bilinear',align_corners=False)[:,0]
                height,width=self._resized_shape
                resized=F.interpolate(resized[:,None,:height,:width],size=bgr.shape[:2],mode='bilinear',align_corners=False)[:,0]
                chosen=resized;binary=chosen>0
                quality=(chosen.sigmoid()*binary).flatten(1).sum(1)/binary.flatten(1).sum(1).clamp_min(1e-6)
                valid=probabilities*quality>self.score
                from detectron2.structures import Instances
                instances=Instances(bgr.shape[:2]);instances.pred_masks=binary[valid]
                instances.pred_classes=self.prompts['sem_prompt_label_list'][0][category[valid]]
                instances.scores=probabilities[valid]
            return self._decode(instances)
        finally:self._last_features=None

    def metadata(self):
        return dict(name='cosine',source_repository='https://github.com/aim-uofa/COSINE',source_revision=self.revision,
            checkpoint_sha256=self.checkpoint_hash,support_bank=self.selection.metadata(),support_cache=str(self.cache),
            ways=len(self.classes),image_size=self.size,clip_image_size=self.clip_size,precision=self.precision,
            confidence_threshold=self.score,mask_nms_iou=self.nms_iou,visual_text_prompts=True,fcclip_scoring=True,
            query_encodings=self.query_encodings,clip_query_encodings=self.clip_query_encodings,
            strict_checkpoint_load=True,restored_shared_tensor_aliases=self.restored_aliases,
            compatibility=['shared published components; missing public FSOD variants bypassed',
                'MMCV equivalent CUDA deformable-attention operator','published LayerNorm2d for old Detectron2',
                'OpenCLIP native text API for version-correct token layout',
                'one CLIP backbone forward shared across pyramid levels'])
