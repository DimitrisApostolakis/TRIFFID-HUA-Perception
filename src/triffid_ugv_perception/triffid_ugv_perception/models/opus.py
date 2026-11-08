"""Official OPUS, joint K-way detection with cached annotated visual prompts.

All learned parameters come from the full released checkpoint. Query annotations
are never used. The official visual.G path averages up to 16 real supports/class.
"""
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
from .pet_dino import support_box


def validate_memory(memory, names, counts, torch):
    expected = [name for name, count in zip(names, counts) for _ in range(count)]
    vectors = memory['embedded']
    if memory['label_names'] != expected or tuple(vectors.shape) != (len(expected), 256):
        raise ValueError('OPUS support cache labels/shape do not match the selected bank')
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError('OPUS support cache contains nonfinite embeddings')


class OpusDetector:
    def __init__(self, config):
        import torch
        self.torch = torch
        self.device = config['device']
        self.score = float(config['confidence_threshold'])
        self.tracking_score = float(config.get('opus_tracking_floor', self.score))
        self.prompt_mode = config.get('opus_prompt_mode', 'visual')
        if self.prompt_mode not in ('visual', 'mixed'):
            raise ValueError('OPUS prompt mode must be visual or mixed')
        self.mode = ('visual' if self.prompt_mode == 'visual' else 'text_visual') + '.G.16'
        self.classes = tuple(classes_for(config['classes']))
        self.names = [item.name for item in self.classes]
        bank = SupportBank(config['support_bank'])
        self.selection = (bank.select_available(self.classes) if config.get('opus_use_all_supports')
                          else bank.select(self.classes, config['shots']))
        self.size = int(config['image_size'])
        if self.size < 32:
            raise ValueError('OPUS image size must be >=32')
        self.half = config['opus_precision'] == 'float16' and self.device.startswith('cuda')
        self.repo = Path(config['opus_repo']).resolve()
        sys.path.insert(0, str(self.repo))
        self.revision = subprocess.check_output(['git', '-C', str(self.repo), 'rev-parse', 'HEAD'], text=True).strip()
        from mmengine.config import Config
        from mmengine.dataset import Compose
        from mmdet.utils import register_all_modules
        register_all_modules()
        from mmdet.registry import MODELS
        cfg = Config.fromfile(str(self.repo / 'configs/opus_dinov3_convnext-b/opus_dinov3_convnext-b_text_visual_pretrain_all.py'))
        cfg.model.backbone.update(model_name=str(self.repo / 'hf_configs/facebook/dinov3-convnext-base-pretrain-lvd1689m'),
                                  use_pretrain=False, train_backbone=False)
        cfg.model.language_model.update(name=str(self.repo / 'hf_configs/openai/clip-vit-base-patch32'),
                                        use_pretrain=False, use_checkpoint=False)
        cfg.model.test_cfg.update(chunked_size=-1)
        self.model = MODELS.build(cfg.model)
        checkpoint = Path(config['opus_checkpoint'])
        self.checkpoint_hash = file_hash(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.to(self.device).eval().requires_grad_(False)
        self.query_encodings = 0
        self.preparing = True
        self.last_features = None
        self.model.backbone.register_forward_hook(self._features)
        # Exactly the native test transforms, after loading the BGR image.
        self.transforms = Compose([
            dict(type='FixScaleResize', scale=(self.size, self.size), keep_ratio=True, backend='pillow'),
            dict(type='Pad', size=(self.size, self.size), pad_val=dict(img=(114, 114, 114))),
            dict(type='PackDetInputs', meta_keys=('img_id', 'ori_shape', 'img_shape', 'scale_factor')),
        ])
        self.counts = [len(self.selection.for_class(item)) for item in self.classes]
        identity = dict(supports=self.selection.metadata(), checkpoint_sha256=self.checkpoint_hash,
                        source_revision=self.revision, image_size=self.size, precision='float16' if self.half else 'float32',
                        schema='triffid.opus.visual-memory.v1', prompt_box_policy='largest_connected_component',
                        support_mode='visual.I.gen_embed', resize='native FixScaleResize/pillow and Pad114')
        directory = Path(config['opus_support_cache'])
        directory.mkdir(parents=True, exist_ok=True)
        self.cache = directory / (hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() + '.pth')
        self.cache_hit = self.cache.exists()
        if self.cache_hit:
            cached = torch.load(self.cache, map_location='cpu', weights_only=False)
            if cached['metadata'] != identity:
                raise ValueError('OPUS cache fingerprint mismatch')
            memory = cached['memory']
        else:
            vectors, labels = [], []
            with torch.inference_mode(), self._amp():
                for index, item in enumerate(self.classes):
                    for support in self.selection.for_class(item):
                        rgb, mask = support.load()
                        data = self._input(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), support_box(mask), index)
                        prepared = self.model.data_preprocessor(data, False)
                        embedded = self.model.predict(prepared['inputs'], prepared['data_samples'])[0].embed_instances
                        if list(embedded.labels) != [index] or list(embedded.label_names) != [item.name]:
                            raise ValueError('OPUS returned a different support class')
                        vectors.append(embedded.embedded[0].detach().float().cpu())
                        labels.append(item.name)
                    print(f'OPUS support memory: {item.name}: {self.counts[index]}', flush=True)
            memory = dict(embedded=torch.stack(vectors), label_names=labels)
            validate_memory(memory, self.names, self.counts, torch)
            torch.save(dict(metadata=identity, memory=memory), self.cache)
        validate_memory(memory, self.names, self.counts, torch)
        self.memory = dict(embedded=memory['embedded'].to(self.device), label_names=memory['label_names'])
        # Install once globally: per-sample metainfo is deep-copied by MMEngine.
        self.model.embed_prompt_memory_global = self.model.load_embed_prompts(self.memory)
        self.last_features = None
        self.preparing = False
        torch.manual_seed(0)

    def _amp(self):
        return self.torch.autocast('cuda', dtype=self.torch.float16) if self.half else contextlib.nullcontext()

    def _features(self, module, inputs, output):
        if not self.preparing:
            self.query_encodings += 1
            self.last_features = output[0].detach()

    def _input(self, bgr, box=None, index=None):
        data = dict(img=bgr, img_id=0, ori_shape=bgr.shape[:2], img_shape=bgr.shape[:2],
                    gt_bboxes=np.array([box] if box is not None else [], dtype=np.float32).reshape(-1, 4),
                    gt_bboxes_labels=np.array([index] if box is not None else [], dtype=np.int64),
                    gt_ignore_flags=np.zeros(1 if box is not None else 0, dtype=bool))
        packed = self.transforms(data)
        sample = packed['data_samples']
        sample.set_metainfo(dict(text=self.names, custom_entities=True,
                                 mode='visual.I.gen_embed' if box is not None else self.mode))
        return dict(inputs=[packed['inputs']], data_samples=[sample])

    def detect(self, bgr):
        torch = self.torch
        with torch.inference_mode(), self._amp():
            data = self.model.data_preprocessor(self._input(bgr), False)
            output = self.model.predict(data['inputs'], data['data_samples'])[0].pred_instances
        result = []
        self.last_candidates, self.last_tracking_detections = [], []
        height, width = bgr.shape[:2]
        scale_x, scale_y = data['data_samples'][0].scale_factor
        padded_h, padded_w = data['inputs'].shape[-2:]
        feature = self.last_features[0].float()
        fh, fw = feature.shape[-2:]
        for box, score, label in zip(output.bboxes.detach().cpu().numpy(),
                                     output.scores.detach().cpu().numpy(), output.labels.detach().cpu().numpy()):
            item = self.classes[int(label)]
            box = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
            if box[2] <= box[0] or box[3] <= box[1] or not np.isfinite(score) or not np.isfinite(box).all():
                continue
            row = dict(bbox=box.tolist(), confidence=float(score), class_id=item.id, class_name=item.name)
            self.last_candidates.append(row)
            if float(score) < min(self.score, self.tracking_score):
                continue
            # Reuse the one query encoding; exclude the letterbox padding.
            x1, y1, x2, y2 = box * np.array([scale_x, scale_y, scale_x, scale_y])
            a, b = max(0, int(x1 / padded_w * fw)), max(0, int(y1 / padded_h * fh))
            c = min(fw, max(a + 1, int(np.ceil(x2 / padded_w * fw))))
            d = min(fh, max(b + 1, int(np.ceil(y2 / padded_h * fh))))
            vector = torch.nn.functional.normalize(feature[:, b:d, a:c].mean((1, 2)), dim=0)
            detection = dict(row, appearance=vector.cpu().numpy().tolist())
            self.last_tracking_detections.append(detection)
            if float(score) >= self.score:
                result.append(detection)
        self.last_features = None
        return result

    def metadata(self):
        return dict(name='opus', backbone='DINOv3 ConvNeXt-B', source_revision=self.revision,
                    checkpoint_sha256=self.checkpoint_hash, strict_checkpoint_load=True,
                    prompt_mode=self.prompt_mode, native_mode=self.mode,
                    text_prompts=['a photo of {}.'.format(name) for name in self.names] if self.prompt_mode == 'mixed' else [],
                    visual_text_fusion='official 0.5*(visual+text)' if self.prompt_mode == 'mixed' else 'visual only',
                    support_bank=self.selection.metadata(), support_cache=str(self.cache), support_cache_hit=self.cache_hit,
                    shots_per_class=dict(zip(self.names, self.counts)),
                    prompt_box_policy='largest_connected_component', query_encodings=self.query_encodings,
                    image_size=self.size, precision='float16' if self.half else 'float32',
                    preprocessing='native FixScaleResize/pillow; aspect preserved; square pad114; ImageNet normalization',
                    query_annotations_used=False, joint_class_count=len(self.classes), random_seed=0)
