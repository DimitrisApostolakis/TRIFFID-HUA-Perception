"""Adapter for the official SANSA universal model, mask-prompted, 1/5 shot.

Uses its public forward() unchanged. Each class/query is an independent episode;
query predictions never become supports for another view. Source reference:
https://github.com/ClaudiaCuttano/SANSA/tree/f562b5642940e08a57e3da2608b804b99af3763b
"""
import contextlib
import importlib
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from .base import ClassPrediction
from ..supports import SupportBank


@contextlib.contextmanager
def _working_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


class SansaBackend:
    def __init__(self, repo, checkpoint, sam2_checkpoint, support_manifest,
                 device='cuda', image_size=640, threshold=0.5):
        self.repo = Path(repo).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.sam2_checkpoint = Path(sam2_checkpoint).resolve()
        for path in (self.repo / 'models/sansa/sansa.py', self.checkpoint, self.sam2_checkpoint):
            if not path.is_file():
                raise ValueError(f'SANSA dependency not found: {path}')
        if image_size < 1 or not 0 < threshold < 1:
            raise ValueError('Invalid SANSA image size or threshold')
        self.bank = SupportBank(support_manifest)
        self.device_name, self.image_size, self.threshold = device, image_size, threshold
        self._prepared = {}
        self.model = None

    def prepare(self, classes):
        # Validate/resize supports before allocating model weights. Cache CPU
        # tensors, not entire full-resolution reference images on the GPU.
        import torch
        self.torch = torch
        if self.device_name.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; select --device cpu or install a CUDA PyTorch build')
        self.device = torch.device(self.device_name)
        for item in classes:
            images, masks = [], []
            for support in self.bank.for_class(item):
                image, mask = support.load()
                images.append(self._image_tensor(image))
                mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                if not mask.any():
                    raise ValueError(f'{item.name}: foreground disappeared when resizing support')
                masks.append(torch.from_numpy(mask.astype(np.float32))[None, None])
            self._prepared[item.id] = (torch.stack(images), masks)
        if self.model is None:
            # Official builder uses relative SAM2 weight paths. Point it at an
            # explicit local checkpoint to prevent surprise weight downloads.
            sys.path.insert(0, str(self.repo))
            with _working_directory(self.repo):
                paths = importlib.import_module('util.path_utils')
                builder = importlib.import_module('models.sansa.sansa')
                if Path(builder.__file__).resolve() != self.repo / 'models/sansa/sansa.py':
                    raise RuntimeError('Another Python models package shadows the SANSA checkout')
                previous = paths.SAM2_PATHS_CONFIG['large']
                paths.SAM2_PATHS_CONFIG['large'] = (str(self.sam2_checkpoint), previous[1])
                try:
                    self.model = builder.build_sansa('large', [2, 3], 0.8, str(self.device))
                finally:
                    paths.SAM2_PATHS_CONFIG['large'] = previous
                state = torch.load(str(self.checkpoint), map_location='cpu', weights_only=True)
                state = state.get('model', state)
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                # Official universal checkpoints may store adapters only, but
                # missing adapter weights would silently produce a random model.
                required = {key for key in self.model.state_dict() if 'adapter' in key}
                if not required or required.intersection(missing) or unexpected:
                    raise ValueError('Checkpoint does not match the SANSA universal architecture')
            self.model.to(self.device).eval()
            self.model.requires_grad_(False)

    def _image_tensor(self, rgb):
        image = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return self.torch.from_numpy(image.transpose(2, 0, 1).copy())

    def predict(self, rgb, classes):
        if self.model is None:
            raise RuntimeError('Call prepare before inference')
        torch = self.torch
        query = self._image_tensor(rgb)
        with torch.inference_mode():
            for item in classes:
                images, masks = self._prepared[item.id]
                # Official prompt layout: batch -> support index -> mask prompt.
                prompt = {'shots': len(masks), 0: {
                    i: {'prompt_type': 'mask', 'prompt': mask.to(self.device)}
                    for i, mask in enumerate(masks)}}
                video = torch.cat((images, query[None]))[None].to(self.device)
                output = self.model(video, prompt)['pred_masks']
                if output.ndim != 3 or output.shape[0] != len(masks) + 1:
                    raise RuntimeError(f'Unexpected official SANSA output: {tuple(output.shape)}')
                logits = torch.nn.functional.interpolate(output[-1:][None], size=rgb.shape[:2],
                                                        mode='bilinear', align_corners=False)[0, 0]
                probabilities = logits.sigmoid().cpu().numpy()
                if not np.isfinite(probabilities).all():
                    raise ValueError('SANSA returned non-finite probabilities')
                # Stream one full-resolution class map at a time to bound CPU
                # memory when working with 4K frames and many classes.
                yield ClassPrediction(item.id, probabilities > self.threshold, probabilities)

    def metadata(self):
        return {'name': 'sansa_universal', 'source_repository': 'https://github.com/ClaudiaCuttano/SANSA',
                'reference_revision': 'f562b5642940e08a57e3da2608b804b99af3763b',
                'checkpoint': self.checkpoint.name, 'sam2_checkpoint': self.sam2_checkpoint.name,
                'image_size': self.image_size, 'threshold': self.threshold,
                'device': self.device_name, 'supports_per_class': {
                    str(key): len(value[1]) for key, value in self._prepared.items()},
                'inference': 'independent one-class support/query episodes',
                'instance_method': '8-connected semantic components; not learned instances'}
