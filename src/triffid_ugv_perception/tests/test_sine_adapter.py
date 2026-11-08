from types import SimpleNamespace
import numpy as np
import pytest
from triffid_ugv_perception.models.sine import SineDetector
from triffid_ugv_perception.ontology import classes_for


def test_sine_masks_keep_original_class_ids_and_exclusive_boxes():
    torch=pytest.importorskip('torch')
    adapter=SineDetector.__new__(SineDetector)
    adapter.by_id={c.id:c for c in classes_for('civilian vehicle,cone')}
    adapter.score=.7;adapter.nms_iou=.5
    adapter._appearance=lambda mask:[1.,0.]
    masks=torch.zeros((3,12,16));masks[0,3:7,5:9]=1;masks[1,0:2,0:2]=1
    output=SimpleNamespace(pred_masks=masks,pred_classes=torch.tensor([21,39,21]),scores=torch.tensor([.9,.2,.95]))
    output.to=lambda device:output
    detections=adapter._decode(output)
    assert len(detections)==1
    d=detections[0]
    assert d['class_id']==21 and d['class_name']=='civilian vehicle'
    assert d['bbox']==(5.,3.,9.,7.)
    assert d['mask'].dtype==np.bool_ and d['mask'].shape==(12,16)
    assert d['mask'].sum()==16
    output.pred_classes[0]=0
    with pytest.raises(ValueError,match='unselected class'):adapter._decode(output)


def test_sine_normalization_precedes_zero_padding():
    torch=pytest.importorskip('torch')
    pytest.importorskip('detectron2')
    adapter=SineDetector.__new__(SineDetector);adapter.torch=torch;adapter.size=14
    adapter.resize=SimpleNamespace(get_transform=lambda image:SimpleNamespace(apply_image=lambda x:x))
    image=np.zeros((2,3,3),np.uint8);image[...,0]=255
    inputs,_,shape=adapter._input(image)
    assert shape==(2,3) and inputs['image'].shape==(3,14,14)
    np.testing.assert_allclose(inputs['image'][:,0,0].numpy(),[-.485/.229,-.456/.224,(1-.406)/.225],rtol=1e-5)
    assert torch.count_nonzero(inputs['image'][:,2:,:]).item()==0
    assert torch.count_nonzero(inputs['image'][:,:,3:]).item()==0
    assert inputs['instances'].image_size==(2,3)


def test_mask_nms_suppresses_duplicate_queries_but_keeps_neighbours():
    from triffid_ugv_perception.models.sine import suppress_overlapping_masks
    left=np.zeros((20,30),bool);left[3:15,2:12]=True
    right=np.zeros((20,30),bool);right[3:15,14:24]=True
    def det(mask,score,cls):
        ys,xs=np.where(mask)
        return dict(mask=mask,confidence=score,class_id=cls,bbox=(xs.min(),ys.min(),xs.max()+1,ys.max()+1))
    a,b,c=det(left,.8,39),det(left.copy(),.9,39),det(right,.75,39)
    selected=suppress_overlapping_masks([a,b,c],.5)
    assert len(selected)==2 and selected[0] is b and selected[1] is c
    # Conflicting class hypotheses on the identical mask must not create
    # separate discoveries of the same region.
    assert len(suppress_overlapping_masks([b,det(left.copy(),.7,21)],.5))==1


def test_sine_channel_layer_norm_matches_detectron2_definition():
    import sys
    from pathlib import Path
    torch=pytest.importorskip('torch')
    root=Path('/mnt/d/Models/ThirdParty/SINE/inference_fsod')
    if not root.exists():pytest.skip('Local SINE checkout unavailable')
    sys.path.insert(0,str(root))
    try:
        from fsdet.model.image_encoder.dinov2 import LayerNorm2d
        norm=LayerNorm2d(8)
        x=torch.arange(2*8*3*4,dtype=torch.float32).reshape(2,8,3,4)/17
        expected=torch.nn.functional.layer_norm(x.permute(0,2,3,1),(8,),norm.weight,norm.bias,eps=1e-6).permute(0,3,1,2)
        torch.testing.assert_close(norm(x),expected,rtol=1e-5,atol=1e-5)
    finally:sys.path.remove(str(root))
