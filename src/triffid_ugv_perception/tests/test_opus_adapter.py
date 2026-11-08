"""Guard support identity and avoid annotation leakage into K-way queries."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from triffid_ugv_perception.models.opus import OpusDetector, validate_memory


def test_memory_requires_exact_class_order_real_shot_counts_and_finite_vectors():
    names, counts = ['citizen', 'vehicle'], [1, 3]
    memory = dict(label_names=['citizen', 'vehicle', 'vehicle', 'vehicle'], embedded=torch.zeros(4, 256))
    validate_memory(memory, names, counts, torch)
    with pytest.raises(ValueError):
        validate_memory(memory, names[::-1], counts, torch)
    with pytest.raises(ValueError):
        validate_memory(memory, names, [2, 2], torch)
    memory['embedded'][1, 0] = float('nan')
    with pytest.raises(ValueError):
        validate_memory(memory, names, counts, torch)


@pytest.mark.parametrize('mode', ['visual.G.16', 'text_visual.G.16'])
def test_k_way_query_has_full_vocabulary_empty_gt_and_no_per_image_filter(mode):
    detector = object.__new__(OpusDetector)
    detector.names = ['citizen', 'first responder', 'vehicle']
    detector.mode = mode
    captured = []
    class Sample:
        def __init__(self): self.metainfo = {}
        def set_metainfo(self, meta): self.metainfo.update(meta)
    def transform(data):
        captured.append(data)
        return dict(inputs=data['img'], data_samples=Sample())
    detector.transforms = transform
    result = detector._input(np.zeros((720, 1280, 3), np.uint8))
    assert captured[0]['gt_bboxes'].shape == (0, 4)
    assert captured[0]['gt_bboxes_labels'].size == 0
    assert result['data_samples'][0].metainfo == dict(text=detector.names, custom_entities=True, mode=mode)
    # A support uses its annotated class index, rather than accidentally class 0.
    support = detector._input(np.zeros((720, 1280, 3), np.uint8), [10, 20, 40, 80], 2)
    assert captured[1]['gt_bboxes_labels'].tolist() == [2]
    assert captured[1]['gt_bboxes'].tolist() == [[10, 20, 40, 80]]
    assert support['data_samples'][0].metainfo['mode'] == 'visual.I.gen_embed'
