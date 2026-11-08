import numpy as np
import pytest
from triffid_ugv_perception.models.pet_dino import support_box


def test_support_box_does_not_enclose_disconnected_instances_and_background():
    mask=np.zeros((100,100),np.uint8)
    mask[10:30,5:15]=1
    mask[80:85,90:95]=1
    assert support_box(mask)==[5,10,15,30]
    with pytest.raises(ValueError):support_box(np.zeros_like(mask))


@pytest.mark.parametrize('mode',['text','visual'])
def test_prompt_mode_routes_class_names_and_visual_memory_separately(monkeypatch,mode):
    import sys
    from types import ModuleType, SimpleNamespace
    import torch
    from triffid_ugv_perception.models.pet_dino import PetDinoDetector
    class Sample:
        def __init__(self,metainfo):self.metainfo=metainfo
        def set_metainfo(self,values):self.metainfo.update(values)
    structures=ModuleType('mmdet.structures');structures.DetDataSample=Sample
    engine=ModuleType('mmengine.structures');engine.InstanceData=SimpleNamespace
    monkeypatch.setitem(sys.modules,'mmdet.structures',structures)
    monkeypatch.setitem(sys.modules,'mmengine.structures',engine)
    detector=object.__new__(PetDinoDetector)
    detector.torch=torch;detector.prompt_mode=mode
    detector.classes=(SimpleNamespace(name='citizen'),SimpleNamespace(name='first responder'))
    detector.size=64;detector.max_size=128;detector.memory=torch.ones(2,256)
    data=detector._input(np.zeros((80,120,3),np.uint8))
    meta=data['data_samples'][0].metainfo
    assert meta['text']==['citizen','first responder'] and meta['custom_entities']
    assert meta['prompt_type']==('Text' if mode=='text' else 'Visual')
    assert ('memory_visual' in meta)==(mode=='visual')
    assert ('memory_label' in meta)==(mode=='visual')
    assert 'prompt_bboxes' not in meta


def test_text_metadata_reports_zero_visual_supports():
    from types import SimpleNamespace
    from triffid_ugv_perception.models.pet_dino import PetDinoDetector
    detector=object.__new__(PetDinoDetector)
    detector.classes=(SimpleNamespace(name='citizen'),)
    detector.prompt_mode='text';detector.selection=None;detector.cache=None
    detector.checkpoint_hash='checkpoint';detector.revision='revision';detector.backbone='swin-t'
    detector.query_encodings=2;detector.size=800;detector.max_size=1333
    meta=detector.metadata()
    assert meta['support_bank'] is None and meta['support_cache'] is None
    assert meta['shots_per_class']=={'citizen':0}
    assert meta['text_prompts']==['citizen'] and meta['prompt_mode']=='text'
