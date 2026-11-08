"""Collage geometry and query-only export; no model weights required."""
from types import SimpleNamespace

import numpy as np
import pytest

from triffid_uav_perception.models.fss_sam3 import FssSam3Backend,one_shot_canvas
from triffid_uav_perception.ontology import BY_ID


def test_collage_box_uses_support_coordinates_and_query_is_at_top():
    support=np.full((8,12,3),(0,0,255),np.uint8)
    query=np.full((6,10,3),(255,0,0),np.uint8)
    mask=np.zeros((8,12),np.uint8);mask[2:6,3:6]=1
    canvas,box,crop=one_shot_canvas(support,mask,query,size=10)
    pixels=np.asarray(canvas)
    assert crop==(0,0,10,4)
    assert (pixels[:4]==[255,0,0]).all() and (pixels[4:]==[0,0,255]).all()
    np.testing.assert_allclose(box,[.375,.7,.25,.3])


def test_support_predictions_cannot_leak_into_query_export():
    torch=pytest.importorskip('torch')
    backend=FssSam3Backend.__new__(FssSam3Backend);backend.torch=torch
    support=np.zeros((8,12,3),np.uint8);mask=np.ones((8,12),np.uint8)
    backend.references={6:(support,mask)}
    masks=torch.zeros((1,1,1008,1008),dtype=torch.bool);masks[:,:,500:600,100:200]=True
    state=dict(masks=masks,masks_logits=masks.float()*.9)
    backend.processor=SimpleNamespace(set_image=lambda image:{},reset_all_prompts=lambda state:None,
        add_geometric_prompt=lambda **kwargs:state)
    prediction=list(backend.predict(np.zeros((24,32,3),np.uint8),[BY_ID[6]]))[0]
    assert prediction.class_id==6 and prediction.mask.shape==(24,32)
    assert not prediction.mask.any() and not prediction.probabilities.any()


def test_text_mode_uses_each_original_role_name_without_merging():
    torch=pytest.importorskip('torch');texts=[]
    backend=FssSam3Backend.__new__(FssSam3Backend)
    backend.torch=torch;backend.use_text=True;backend.diagnostics=[]
    backend.references={i:(np.zeros((8,12,3),np.uint8),np.ones((8,12),np.uint8)) for i in (6,23)}
    result=dict(masks=torch.zeros((0,1,1008,1008),dtype=torch.bool),scores=torch.empty(0))
    def text_prompt(name,state):texts.append(name);return state
    backend.processor=SimpleNamespace(set_image=lambda image:{},reset_all_prompts=lambda state:None,
        set_text_prompt=text_prompt,add_geometric_prompt=lambda **kwargs:result)
    predictions=list(backend.predict(np.zeros((24,32,3),np.uint8),[BY_ID[6],BY_ID[23]]))
    assert texts==['first responder','citizen']
    assert [p.class_id for p in predictions]==[6,23]
