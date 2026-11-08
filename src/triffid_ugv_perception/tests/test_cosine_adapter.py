from types import SimpleNamespace
import numpy as np
import torch

from triffid_ugv_perception.models.cosine import CosineDetector, single_clip_forward


def test_clip_pyramid_calls_backbone_once_and_keeps_all_levels():
    calls=[]
    values={name:torch.tensor([i]) for i,name in enumerate(['res2','res3','res4','res5','clip_vis_dense'])}
    def forward(pixels):calls.append(pixels);return values
    wrapper=SimpleNamespace(encoder=forward,feature_layer_list=list(values))
    pixels=torch.zeros(1,3,32,32)
    output=single_clip_forward(wrapper,pixels)
    assert len(calls)==1
    for actual,expected in zip(output,values.values()):assert actual is expected


def test_preprocessing_uses_rgb_separate_normalizations_and_valid_padding():
    from detectron2.data.transforms import ResizeShortestEdge
    detector=object.__new__(CosineDetector)
    detector.torch=torch;detector.size=56;detector.clip_size=64
    detector.resize=ResizeShortestEdge(56,56)
    bgr=np.zeros((28,56,3),np.uint8);bgr[:,:,2]=255
    inputs,_,shape=detector._input(bgr)
    assert shape==(28,56)
    assert inputs['image'].shape==(3,56,56)
    assert inputs['clip_image'].shape==(3,64,64)
    assert inputs['image'][:,28:].count_nonzero()==0
    assert inputs['clip_image'][:,32:].count_nonzero()==0
    assert torch.allclose(inputs['image'][:,0,0],torch.tensor([(1-.485)/.229,-.456/.224,-.406/.225]))
    expected=torch.tensor([(255-122.7709383)/68.5005327,-116.7460125/66.6321579,-104.09373615/70.32316305])
    assert torch.allclose(inputs['clip_image'][:,0,0],expected,atol=1e-5)
