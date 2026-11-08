import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from triffid_ugv_perception.ontology import BY_NAME
from triffid_ugv_perception.supports import SupportBank, bank_template

spec=importlib.util.spec_from_file_location('ls_import',Path(__file__).resolve().parents[3]/'scripts/import_labelstudio_bank.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def literal_rle(values):
    # Independent literal stream in the documented rle-pack format.
    bits=f'{len(values):032b}00111'+'0010001101111111'
    for value in values:bits+='000000'+f'{value:08b}'
    bits+='0'*((-len(bits))%8)
    return [int(bits[i:i+8],2) for i in range(0,len(bits),8)]


def test_rgba_alpha_and_malformed_dimensions():
    rgba=np.zeros((2,2,4),np.uint8)
    rgba[:,:,0]=255  # RGB values alone are not foreground.
    rgba[0,1,3]=255
    actual=module.decode_brush(literal_rle(rgba.ravel()),2,2)
    assert actual.tolist()==[[False,True],[False,False]]
    with pytest.raises(ValueError,match='dimensions'):
        module.decode_brush(literal_rle(rgba.ravel()),3,2)
    with pytest.raises(ValueError,match='Truncated'):
        module.decode_brush([0],2,2)


def test_exclusions_override_export_and_overlapping_classes_survive(tmp_path):
    root=tmp_path/'bank';source=root/'source_annotations'
    (source/'images').mkdir(parents=True)
    cv2.imwrite(str(source/'images/example.png'),np.zeros((2,2,3),np.uint8))
    (source/'manifest.json').write_text(json.dumps(dict(items=[dict(task_id='5',image='images/example.png')])))
    original=bank_template();(root/'bank.json').write_text(json.dumps(original))
    def region(name):
        return dict(id=name,type='brushlabels',original_width=2,original_height=2,
                    value=dict(brushlabels=[name],rle=literal_rle([255]*16)))
    tasks=[dict(id=5,data=dict(image='/data/upload/4/example.png'),annotations=[dict(result=[region('water'),region('fence')])]),
           dict(id=29,data=dict(image='absent-uav.png'),annotations=[])]
    export=tmp_path/'export.json';export.write_text(json.dumps(tasks))
    report=module.import_export(export,root,[29,31,33,34])
    assert report['included_task_ids']==[5]
    assert report['excluded_ids_present_in_export']==[29]
    assert report['one_shot_classes']==['water','fence']
    selected=SupportBank(root).select([BY_NAME['water'],BY_NAME['fence']],shots=1)
    for _,rows in selected.examples:assert rows[0].load()[1].all()
    archived=next((root/'history').glob('*/bank.json'))
    assert json.loads(archived.read_text())==original


def test_new_raw_photograph_is_copied_into_portable_bank(tmp_path):
    root=tmp_path/'bank';source=root/'source_annotations';source.mkdir(parents=True)
    (source/'manifest.json').write_text(json.dumps(dict(items=[])))
    raw=tmp_path/'raw';raw.mkdir()
    image=raw/'IMG_NEW.jpg';cv2.imwrite(str(image),np.zeros((2,2,3),np.uint8))
    before=image.read_bytes()
    region=dict(id='person',type='brushlabels',original_width=2,original_height=2,
                value=dict(brushlabels=['citizen'],rle=literal_rle([255]*16)))
    export=tmp_path/'export.json';export.write_text(json.dumps([dict(id=43,
        data=dict(image='/data/upload/4/abcdef12-IMG_NEW.jpg'),annotations=[dict(result=[region])])]))
    result=module.import_export(export,root,[],raw)
    assert result['classes']['citizen']['imported']==1
    support=SupportBank(root).select([BY_NAME['citizen']],1).for_class(BY_NAME['citizen'])[0]
    assert support.image.is_relative_to(root)
    assert support.image.read_bytes()==before==image.read_bytes()
    # Later exports must resolve the managed copy without the old external directory.
    image.unlink()
    tasks=json.loads(export.read_text());tasks[0]['annotations'][0]['result'][0]['id']='edited-person'
    second=tmp_path/'second.json';second.write_text(json.dumps(tasks))
    assert module.import_export(second,root,[])['classes']['citizen']['imported']==1


def test_pixel_rle_under_rectangle_widget_and_box_mask_consistency(tmp_path):
    root=tmp_path/'bank';source=root/'source_annotations';(source/'images').mkdir(parents=True)
    cv2.imwrite(str(source/'images/frame.png'),np.zeros((20,20,3),np.uint8))
    (source/'manifest.json').write_text(json.dumps(dict(items=[dict(task_id='49',image='images/frame.png')])))
    def region(name,mask,kind='brushlabels'):
        rgba=np.zeros((20,20,4),np.uint8);rgba[:,:,3]=mask*255
        return dict(id=name,type=kind,original_width=20,original_height=20,
                    value={kind:[name],'rle':literal_rle(rgba.ravel())})
    small=np.zeros((20,20),np.uint8);small[5:9,5:9]=1
    valid=region('boot',small,'rectanglelabels')
    invalid=region('boot',np.ones((20,20),np.uint8));invalid['id']='wrong-car-mask'
    box=dict(id='box',type='rectanglelabels',value=dict(x=25,y=25,width=20,height=20,rectanglelabels=['boot']))
    export=tmp_path/'export.json';export.write_text(json.dumps([dict(id=49,data=dict(image='/frame.png'),
        annotations=[dict(result=[valid,invalid,box])])]))
    report=module.import_export(export,root,[],validate_box_tasks=[49])
    assert report['classes']['boot']['imported']==1
    assert report['skipped_regions'][0]['region_id']=='wrong-car-mask'
    support=SupportBank(root).select([BY_NAME['boot']],1).for_class(BY_NAME['boot'])[0]
    assert support.load()[1].sum()==16
