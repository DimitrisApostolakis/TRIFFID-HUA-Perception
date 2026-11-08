"""Official FoRIS/FROST inference with local frozen DINOv3 weights.

Only model construction changes the machine-specific hub/checkpoint paths.
The authors' segment(), preprocessing and postprocessing remain unchanged.
"""
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from .base import ClassPrediction


class Dinov3FewShotBackend:
    def __init__(self, method, repo, dino_repo, checkpoint, *, image_size=1024, device='cuda'):
        if method not in ('foris', 'frost', 'insid3'):
            raise ValueError('Expected foris, frost or insid3')
        self.method, self.repo = method, Path(repo).resolve()
        self.dino_repo, self.checkpoint = Path(dino_repo).resolve(), Path(checkpoint).resolve()
        self.image_size, self.device = image_size, device
        self.model, self.references = None, {}

    def prepare(self, classes, supports):
        import torch
        supports.require_classes(classes)
        self.references = {c.id: [support.load() for support in supports.for_class(c)] for c in classes}
        raw = torch.hub.load(str(self.dino_repo), 'dinov3_vitl16', source='local',
                             weights=str(self.checkpoint)).to(self.device).eval().requires_grad_(False)
        sys.path.insert(0, str(self.repo))
        if self.method == 'foris':
            from models.foris import FoRIS
            self.model = FoRIS(raw, image_size=self.image_size, device=self.device,
                               svd_components=500, tau=.6, mask_refiner='bilinear',
                               resize_to_orig_size=True).eval().requires_grad_(False)
        elif self.method == 'insid3':
            from models.insid3 import INSID3
            self.model = INSID3(raw, image_size=self.image_size, device=self.device,
                                mask_refiner='bilinear', resize_to_orig_size=True).eval().requires_grad_(False)
        else:
            from frost import FROST, DINOv3FeatureExtractor
            features = DINOv3FeatureExtractor(raw, image_size=self.image_size, patch_size=16)
            self.model = FROST(features, raw_encoder=raw, image_size=self.image_size,
                               device=self.device, resize_to_orig_size=True).to(self.device).eval().requires_grad_(False)

    def predict(self, rgb, classes):
        import torch
        query = Image.fromarray(rgb)
        with torch.inference_mode():
            for item in classes:
                for image, mask in self.references[item.id]:
                    self.model.set_reference(Image.fromarray(image), Image.fromarray(mask.astype(np.uint8)*255))
                self.model.set_target(query)
                mask = self.model.segment().detach().cpu().numpy().astype(bool)
                if mask.shape != rgb.shape[:2]:
                    raise RuntimeError(f'{self.method}: wrong output dimensions {mask.shape}')
                yield ClassPrediction(item.id, mask, None)

    def metadata(self):
        return dict(name=self.method, backbone='DINOv3 ViT-L/16 LVD-1689M',
                    checkpoint=str(self.checkpoint), device=self.device, image_size=self.image_size,
                    reference_revision=subprocess.check_output(['git', '-C', str(self.repo), 'rev-parse', 'HEAD'], text=True).strip(),
                    supports_per_class={str(cid): len(items) for cid, items in self.references.items()},
                    encoder_source='official local DINOv3 hub; frozen',
                    inference='independent binary class episodes using official segment()',
                    refinement='bilinear (no optional CRF)' if self.method in ('foris','insid3') else 'official FROST spatial refinement',
                    probabilities='not exposed; native binary mask only')
