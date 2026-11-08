"""Official RSRM/FSSDINO algorithms using the user's available DINOv3-L weights."""
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
from .base import ClassPrediction

class DinoVariantBackend:
    def __init__(self,method):
        self.method=method
        self.root=Path(__file__).resolve().parents[4]
        self.repo=self.root/'uav_models'/('RSRM' if method=='rsrm' else 'FSSDINO')
        self.checkpoint=Path('/mnt/d/Models/Checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
        self.size=480 if method=='rsrm' else 512

    def prepare(self,classes,supports):
        import torch
        sys.path.insert(0,str(self.repo))
        self.references={c.id:supports.for_class(c)[0].load() for c in classes}
        if self.method=='rsrm':
            from model.rsrm import RSRM
            # RSRM's local DINOv3 fork also exposes attention features. Only
            # substitute the available encoder size/path, preserving forward().
            load=torch.hub.load
            def local_encoder(*args,**kwargs):
                return load(str(self.repo/'dinov3'),'dinov3_vitl16',source='local',weights=str(self.checkpoint))
            try:
                torch.hub.load=local_encoder
                self.model=RSRM('dinov3')
            finally:torch.hub.load=load
            self.model.layers=len(self.model.backbone.blocks)
            self.model.cuda().eval().requires_grad_(False)
        else:
            from fssdino.module import FSSModule
            raw=torch.hub.load('/mnt/d/Models/ThirdParty/dinov3','dinov3_vitl16',source='local',weights=str(self.checkpoint))
            self.encoder=raw.cuda().eval().requires_grad_(False)
            self.model=FSSModule([{'id':0},{'id':1}],[0,1],raw,torch.device('cuda'))

    def tensor(self,rgb):
        from PIL import Image
        from torchvision import transforms
        transform=transforms.Compose([transforms.Resize((self.size,self.size)),transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        return transform(Image.fromarray(rgb))[None].cuda()

    def predict(self,rgb,classes):
        import torch
        import torch.nn.functional as F
        with torch.inference_mode():
            query=self.tensor(rgb)
            for c in classes:
                image,mask=self.references[c.id];reference=self.tensor(image)
                foreground=torch.from_numpy(cv2.resize(mask,(self.size,self.size),interpolation=cv2.INTER_NEAREST).astype(np.float32))[None].cuda()
                if not foreground.any():raise ValueError('Foreground lost in support resizing')
                if self.method=='rsrm':
                    logits=self.model([reference],[foreground],query)[0]
                    prediction=logits.argmax(1)[0]
                    prediction=F.interpolate(prediction[None,None].float(),size=rgb.shape[:2],mode='nearest')[0,0]
                else:
                    from fssdino.utils import align_masks_to_features
                    prefix=1+self.encoder.n_storage_tokens
                    features=self.encoder.forward_features(torch.cat([query,reference]))['x_prenorm'][:,prefix:]
                    masks=torch.stack([1-foreground,foreground],dim=1)
                    sf,sm=align_masks_to_features(features[1:],masks)
                    side=int(features.shape[1]**.5)
                    qf=features[:1].permute(0,2,1).reshape(1,-1,side,side)
                    prototypes=self.model.cluster_support_image_features_external(sf,sm,n_clusters=5)
                    similarities=self.model.calculate_sim_maps(prototypes,qf)
                    similarities=self.model.add_gram_refinement(similarities,sf,sm,qf)
                    proposals,_=self.model.class_region_proposal(similarities,self.size)
                    stacked=torch.stack([v for v in proposals.values()])[None].float()
                    prediction=F.interpolate(stacked,size=rgb.shape[:2],mode='nearest-exact')[0].argmax(0)
                yield ClassPrediction(c.id,prediction.detach().cpu().numpy().astype(bool))

    def metadata(self):
        return dict(name=self.method,backbone='DINOv3 ViT-L/16 LVD-1689M',
            checkpoint=str(self.checkpoint),image_size=self.size,
            source_revision=subprocess.check_output(['git','-C',str(self.repo),'rev-parse','HEAD'],text=True).strip(),
            variant='User-authorized ViT-L replacement for released ViT-B configuration',
            layers=24,last_layer_only=self.method=='fssdino',
            inference='official RSRM.forward' if self.method=='rsrm' else 'official FSSDINO prototype, similarity, Gram refinement and class-region proposal',
            fssdino_feature_policy='last block pre-final-normalization tokens, matching HF hidden_states[-1]; class/register tokens removed',
            n_clusters=5 if self.method=='fssdino' else None)
