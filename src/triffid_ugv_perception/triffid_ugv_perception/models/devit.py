"""Adapter for the released DE-ViT Small detector; no target-class fine-tuning.

Support extraction follows the authors' demo/build_prototypes.ipynb: masked
DINOv2 patch pooling, followed by class averaging/normalization. Inference uses
the authors' detector unchanged. The checkpoint's COCO class buffers are replaced
by the selected UGV supports; learned detector weights must all match.
"""
import hashlib
from pathlib import Path
import sys
import subprocess
import cv2
import numpy as np
from ..ontology import classes_for
from ..supports import SupportBank


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class DevitDetector:
    def __init__(self, config):
        import torch
        self.torch = torch
        self.classes = tuple(classes_for(config['classes']))
        if len(self.classes) < 2:
            raise ValueError('This DE-ViT baseline requires at least two selected classes')
        self.selection = SupportBank(config['support_bank']).select(self.classes, config['shots'])
        self.repo = Path(config['devit_repo']).resolve()
        checkpoint = Path(config['devit_checkpoint']).resolve()
        background = Path(config['devit_background']).resolve()
        for path in (self.repo/'detectron2/modeling/meta_arch/devit.py', checkpoint, background):
            if not path.is_file():
                raise ValueError(f'DE-ViT dependency missing: {path}')
        self.device = str(config['device'])
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; choose device=cpu explicitly')
        sys.path.insert(0,str(self.repo))
        from detectron2.config import get_cfg
        from detectron2.modeling import build_backbone
        from detectron2.modeling.meta_arch.devit import OpenSetDetectorWithExamples
        from detectron2.data.transforms import ResizeShortestEdge
        import detectron2
        if Path(detectron2.__file__).resolve().parent != self.repo/'detectron2':
            raise RuntimeError('DE-ViT requires its vendored Detectron2')
        cfg = get_cfg()
        cfg.merge_from_file(str(self.repo/'configs/few-shot/vits_shot5.yaml'))
        cfg.MODEL.DEVICE = self.device
        cfg.MODEL.MASK_ON = False
        cfg.DE.BG_PROTOTYPES = str(background)
        cfg.DE.OFFLINE_RPN_CONFIG = str(self.repo/'configs/RPN/mask_rcnn_R_50_C4_1x_fewshot_14.yaml')
        cfg.DE.OFFLINE_RPN_POST_NMS_TOPK_TEST = int(config['devit_proposals'])
        cfg.DE.TOPK = min(10,len(self.classes))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(config['confidence_threshold'])
        cfg.INPUT.MIN_SIZE_TEST = int(config['image_size'])
        cfg.INPUT.MAX_SIZE_TEST = int(config['devit_max_size'])
        if cfg.DE.OFFLINE_RPN_POST_NMS_TOPK_TEST < 1 or cfg.INPUT.MIN_SIZE_TEST < 28 or cfg.INPUT.MAX_SIZE_TEST < cfg.INPUT.MIN_SIZE_TEST:
            raise ValueError('Invalid DE-ViT proposal budget or image dimensions')
        self.resize = ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST,cfg.INPUT.MAX_SIZE_TEST)
        self.support_resize = ResizeShortestEdge(800,1333)
        self.checkpoint_hash = file_hash(checkpoint)
        self.revision = subprocess.check_output(['git','-C',str(self.repo),'rev-parse','HEAD'],text=True).strip()
        identity = dict(supports=self.selection.metadata(), checkpoint_sha256=self.checkpoint_hash,source_revision=self.revision,
                        support_preprocessing='official demo RGB ImageNet norm; shortest=800,max=1333; nearest-14 bicubic',
                        schema='triffid.devit.supports.v1')
        self.cache = Path(config['devit_support_cache']).resolve()
        self.cache.mkdir(parents=True,exist_ok=True)
        import json
        cache_key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        proto_file = self.cache/(cache_key+'.pth')
        state = torch.load(checkpoint,map_location='cpu')['model']
        if not proto_file.is_file():
            backbone = build_backbone(cfg)
            backbone.load_state_dict({k[len('backbone.'):]:v for k,v in state.items() if k.startswith('backbone.')},strict=True)
            backbone.to(self.device).eval().requires_grad_(False)
            prototypes = self._build_prototypes(backbone)
            torch.save(dict(prototypes=prototypes,label_names=[c.name for c in self.classes],metadata=identity),proto_file)
            del backbone
        prototype_data = torch.load(proto_file,map_location='cpu')
        if prototype_data.get('metadata') != identity:
            raise ValueError('Support cache metadata does not match this run')
        cfg.DE.CLASS_PROTOTYPES = str(proto_file)
        params = OpenSetDetectorWithExamples.from_config(cfg)
        params['seen_cids'] = params['all_cids'] = [c.name for c in self.classes]
        self.model = OpenSetDetectorWithExamples(**params)
        # These are data buffers, not learned class-specific parameters. Keep our
        # bank's buffers instead of overwriting them with the checkpoint's labels.
        class_buffers = {'train_class_weight','test_class_weight'}
        incompatible = self.model.load_state_dict({k:v for k,v in state.items() if k not in class_buffers},strict=False)
        if set(incompatible.missing_keys) != class_buffers or incompatible.unexpected_keys:
            raise ValueError(f'Incompatible released DE-ViT checkpoint: {incompatible}')
        del state
        self.model.to(self.device).eval().requires_grad_(False)
        self.cfg = cfg
        self.score = float(config['confidence_threshold'])
        self.nms_iou = float(config['devit_nms_iou'])
        self.min_box_area = float(config['devit_min_box_area'])
        if not 0<self.nms_iou<=1 or self.min_box_area<0:
            raise ValueError('Invalid DE-ViT post-processing parameters')
        self.prototype_file = proto_file
        self._last_features = None
        self._resized_shape = None
        self.query_encodings = 0
        # Retain already-computed query features for appearance descriptors; no
        # second encoder or changes to the detector's forward implementation.
        self._feature_hook = self.model.backbone.register_forward_hook(self._capture_features)

    def _capture_features(self, module, inputs, output):
        self._last_features = output[self.model.vit_feat_name].detach()
        self.query_encodings += 1

    def _build_prototypes(self, backbone):
        torch = self.torch
        from torchvision.transforms import functional as tf
        from torchvision.transforms import InterpolationMode
        encoded, pooled = {}, []
        with torch.inference_mode():
            for item in self.classes:
                vectors = []
                for support in self.selection.for_class(item):
                    if support.image not in encoded:
                        rgb = cv2.cvtColor(cv2.imread(str(support.image)),cv2.COLOR_BGR2RGB)
                        transform = self.support_resize.get_transform(rgb)
                        image = torch.as_tensor(transform.apply_image(rgb).copy()).permute(2,0,1)
                        shape = tuple(max(round(v/14),1)*14 for v in image.shape[1:])
                        image = tf.resize(image,shape,interpolation=InterpolationMode.BICUBIC).float()
                        image = (image-torch.tensor([123.675,116.280,103.530])[:,None,None])/torch.tensor([58.395,57.120,57.375])[:,None,None]
                        tokens = backbone.get_intermediate_layers(image[None].to(self.device),return_class_token=True,reshape=True)[0][0][0].cpu()
                        encoded[support.image] = (transform,tokens)
                    transform,tokens = encoded[support.image]
                    raw = cv2.imread(str(support.mask),cv2.IMREAD_UNCHANGED)
                    binary = raw>0 if support.foreground_value is None else raw==support.foreground_value
                    resized = transform.apply_segmentation(binary.astype(np.uint8))
                    weights = tf.resize(torch.from_numpy(resized.copy())[None].float(),tokens.shape[-2:])
                    if weights.sum() <= .5:
                        raise ValueError(f'{support.example_id}: too little support foreground on DE-ViT feature grid')
                    vectors.append((tokens*weights).flatten(1).sum(1)/weights.sum())
                pooled.append(torch.stack(vectors).mean(0))
        return torch.nn.functional.normalize(torch.stack(pooled),dim=-1)

    def detect(self, bgr):
        torch = self.torch
        h,w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        image = self.resize.get_transform(rgb).apply_image(rgb)
        self._resized_shape = image.shape[:2]
        batch = dict(height=h,width=w,image=torch.as_tensor(np.ascontiguousarray(image.transpose(2,0,1))).to(self.device))
        with torch.inference_mode():
            output = self.model([batch])
        if not output:
            return []
        instances = output[0]['instances'].to('cpu')
        detections = []
        for box,cls,score in zip(instances.pred_boxes.tensor.numpy(),instances.pred_classes.numpy(),instances.scores.numpy()):
            if not np.isfinite(score) or score < self.score:
                continue
            item = self.classes[int(cls)]
            x1,y1,x2,y2 = box.astype(float)
            x1,x2 = np.clip([x1,x2],0,w); y1,y2 = np.clip([y1,y2],0,h)
            if x2<=x1 or y2<=y1:
                continue
            detections.append(dict(bbox=(float(x1),float(y1),float(x2),float(y2)),class_id=item.id,
                class_name=item.name,confidence=float(score),mask=None))
        # Match the authors' demo's area filter + class-agnostic NMS, so
        # overlapping class hypotheses do not all become separate objects.
        if detections:
            from torchvision.ops import nms, box_area
            boxes=torch.tensor([d['bbox'] for d in detections],dtype=torch.float32)
            scores=torch.tensor([d['confidence'] for d in detections],dtype=torch.float32)
            eligible=torch.where(box_area(boxes)>=self.min_box_area)[0]
            keep=eligible[nms(boxes[eligible],scores[eligible],self.nms_iou)].tolist()
            detections=[detections[index] for index in keep]
            for detection in detections:
                detection['appearance']=self._appearance(detection['bbox'],(h,w))
        self._last_features = None
        return detections

    def _appearance(self, box, shape):
        # Mean ROI descriptor from the frozen DINOv2 features for registry ReID.
        torch = self.torch
        features = self._last_features[0]
        ih,iw = self._resized_shape
        h,w = shape
        x1,y1,x2,y2 = box
        ax = max(0,int(x1/w*iw/14)); bx = min(features.shape[2],max(ax+1,int(np.ceil(x2/w*iw/14))))
        ay = max(0,int(y1/h*ih/14)); by = min(features.shape[1],max(ay+1,int(np.ceil(y2/h*ih/14))))
        vector = features[:,ay:by,ax:bx].float().mean((1,2))
        return torch.nn.functional.normalize(vector,dim=0).cpu().numpy().tolist()

    def metadata(self):
        return dict(name='devit',source_repository='https://github.com/mlzxy/devit',
            checkpoint_sha256=self.checkpoint_hash,source_revision=self.revision,backbone='ViT-S/14',ways=len(self.classes),
            support_bank=self.selection.metadata(),support_cache=str(self.prototype_file),
            image_size=self.cfg.INPUT.MIN_SIZE_TEST,max_size=self.cfg.INPUT.MAX_SIZE_TEST,
            proposals=self.cfg.DE.OFFLINE_RPN_POST_NMS_TOPK_TEST,classes_per_proposal=self.cfg.DE.TOPK,
            instance_method='official DE-ViT region propagation boxes; mask head disabled',
            class_agnostic_nms_iou=self.nms_iou,min_box_area=self.min_box_area,confidence_threshold=self.score,
            score_type='released detector score; uncalibrated on PUI',query_encodings=self.query_encodings)
