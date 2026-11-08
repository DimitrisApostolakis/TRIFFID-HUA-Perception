import json
import cv2
import numpy as np
import pytest
from triffid_ugv_perception.ontology import classes_for
from triffid_ugv_perception.supports import SupportBank, initialize_bank, add_example, expand_bank


def test_sixteen_distinct_examples_and_intermediate_shots(tmp_path):
    root=tmp_path/'bank';initialize_bank(root)
    image=tmp_path/'image.png';mask=tmp_path/'mask.png'
    cv2.imwrite(str(mask),np.full((8,9),255,np.uint8))
    for slot in range(1,17):
        cv2.imwrite(str(image),np.full((8,9,3),slot,np.uint8))
        add_example(root,'cone',slot,image,mask)
    bank=SupportBank(root);classes=classes_for('cone')
    for shots in (1,3,5,6,16):
        selected=bank.select(classes,shots)
        assert len(selected.for_class(classes[0]))==shots
        assert selected.metadata()['examples_per_class']==16
    assert bank.validate(classes)['valid']
    available=bank.select_available(classes)
    assert available.shots=='available' and len(available.for_class(classes[0]))==16
    for shots in (0,17,True,3.5):
        with pytest.raises(ValueError):bank.select(classes,shots)


def test_expand_preserves_legacy_and_missing_slots_fail(tmp_path):
    root=tmp_path/'bank';path=initialize_bank(root)
    data=json.loads(path.read_text());data['examples_per_class']=5
    data['classes']={name:entries[:5] for name,entries in data['classes'].items()}
    path.write_text(json.dumps(data));original=path.read_bytes()
    legacy=SupportBank(root)
    with pytest.raises(ValueError,match='capacity'):legacy.select(classes_for('cone'),16)
    expand_bank(root)
    updated=json.loads(path.read_text())
    assert all(updated['classes'][name][:5]==entries for name,entries in data['classes'].items())
    assert next((root/'history').glob('*/bank.json')).read_bytes()==original
    before=path.read_bytes();expand_bank(root);assert path.read_bytes()==before
    with pytest.raises(ValueError,match='Invalid support bank'):SupportBank(root).select(classes_for('cone'),16)
    assert SupportBank(root).validate(classes_for('cone'),shots=3)['expected_examples']==3
