"""FSS-SAM3 one-shot visual-only collage baseline with local SAM3 weights.

Matches model/evaluate.py's COCO 1-shot layout: 1008-square canvas,
query at top (40%), support at bottom (60%), positive support bbox.
"""
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

from .base import ClassPrediction


def one_shot_canvas(support_rgb,support_mask,query_rgb,size=1008,ratio=.6):
    ys,xs=np.nonzero(support_mask)
    if not len(xs):raise ValueError('Empty visual support mask')
    support_h=int(size*ratio);query_h=size-support_h
    canvas=Image.new('RGB',(size,size))
    canvas.paste(Image.fromarray(query_rgb).resize((size,query_h),Image.Resampling.BILINEAR),(0,0))
    canvas.paste(Image.fromarray(support_rgb).resize((size,support_h),Image.Resampling.BILINEAR),(0,query_h))
    height,width=support_mask.shape
    x0,x1=int(xs.min()),int(xs.max()+1);y0,y1=int(ys.min()),int(ys.max()+1)
    box=[(x0+x1)/2/width,(query_h+(y0+y1)/2/height*support_h)/size,
         (x1-x0)/width,(y1-y0)/height*support_h/size]
    return canvas,box,(0,0,size,query_h)


class FssSam3Backend:
    def __init__(self,repo,checkpoint,bpe_path,device='cuda',threshold=.5,use_text=False):
        self.repo=Path(repo).resolve();self.checkpoint=Path(checkpoint).resolve()
        self.bpe_path=Path(bpe_path).resolve();self.device=device;self.threshold=threshold
        self.use_text=use_text;self.diagnostics=[]
        self.model=None;self.references={}
        for p in (self.repo/'sam3/model_builder.py',self.checkpoint,self.bpe_path):
            if not p.is_file():raise ValueError(f'Missing FSS-SAM3 asset: {p}')

    def prepare(self,classes,supports):
        import torch
        supports.require_classes(classes)
        if supports.shots!=1:raise ValueError('This baseline implements the official one-shot layout only')
        self.torch=torch
        self.references={c.id:supports.for_class(c)[0].load() for c in classes}
        sys.path.insert(0,str(self.repo))
        import sam3
        from sam3.model.sam3_image_processor import Sam3Processor
        if Path(sam3.__file__).resolve().parent!=self.repo/'sam3':
            raise RuntimeError('Another SAM3 installation shadows the selected FSS-SAM3 source')
        self.model=sam3.build_sam3_image_model(bpe_path=str(self.bpe_path),device=self.device,
            checkpoint_path=str(self.checkpoint),load_from_HF=False,eval_mode=True,compile=False)
        self.model.eval().requires_grad_(False)
        self.processor=Sam3Processor(self.model,resolution=1008,device=self.device,confidence_threshold=self.threshold)

    def predict(self,rgb,classes):
        torch=self.torch
        with torch.inference_mode():
            for item in classes:
                support,mask=self.references[item.id]
                canvas,box,(x,y,w,h)=one_shot_canvas(support,mask,rgb)
                state=self.processor.set_image(canvas)
                self.processor.reset_all_prompts(state)
                if getattr(self,'use_text',False):state=self.processor.set_text_prompt(item.name,state)
                state=self.processor.add_geometric_prompt(state=state,box=box,label=True)
                if hasattr(self,'diagnostics'):
                    self.diagnostics.append(dict(class_name=item.name,accepted_canvas_masks=len(state['masks']),
                        max_canvas_score=float(state['scores'].max().item()) if len(state['scores']) else None,
                        query_foreground_pixels=int(state['masks'][:,:,y:y+h,x:x+w].sum().item())))
                if len(state['masks']):
                    union=state['masks'][:,0].any(0)[y:y+h,x:x+w].cpu().numpy().astype(np.uint8)
                    probability=state['masks_logits'][:,0].max(0).values[y:y+h,x:x+w].float().cpu().numpy()
                else:
                    union=np.zeros((h,w),np.uint8);probability=np.zeros((h,w),np.float32)
                shape=(rgb.shape[1],rgb.shape[0])
                restored=cv2.resize(union,shape,interpolation=cv2.INTER_NEAREST).astype(bool)
                probability=cv2.resize(probability,shape,interpolation=cv2.INTER_NEAREST)
                yield ClassPrediction(item.id,restored,probability)

    def metadata(self):
        return dict(name='fss-sam3',source_repository='https://github.com/WongKinYiu/FSS-SAM3',
            reference_revision=subprocess.check_output(['git','-C',str(self.repo),'rev-parse','HEAD'],text=True).strip(),
            checkpoint=self.checkpoint.name,device=self.device,precision='float32',
            supports_per_class={str(k):1 for k in self.references},canvas_size=1008,
            support_fraction=.6,support_position='bottom',query_position='top',
            prompting='class-name text plus visual positive support box' if self.use_text else 'visual positive support box; no class-name text; native dummy text visual',
            native_canvas_diagnostics=self.diagnostics,
            object_score_threshold=self.threshold,mask_threshold=.5,
            support_box='tight union bbox of the same original annotated support mask',
            query_restore='crop query from canvas; nearest-neighbour resize to original RGB dimensions',
            inference='independent one-class support/query collage episodes',
            instance_method='official FSS mask union followed by native UAV 8-connected components')
