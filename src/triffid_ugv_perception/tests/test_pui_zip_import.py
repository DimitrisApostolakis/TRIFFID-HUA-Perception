"""Portable support import preserves annotations, distinct-image selection and prior bank."""
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

SCRIPT=Path('/mnt/d/projects/triffid-hua-perception/scripts/import_pui_zip.py')
spec=importlib.util.spec_from_file_location('pui_zip_import',SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def bundle(tmp_path,bad=False):
    image=np.zeros((8,10,3),np.uint8);image[2:5,3:7]=100
    mask=np.zeros((8,10),np.uint8);mask[2:5,3:7]=127 if bad else 255
    image_bytes=cv2.imencode('.png',image)[1].tobytes()
    mask_bytes=cv2.imencode('.png',mask)[1].tobytes()
    data=dict(images=[dict(id=i,file_name=f'images/{i}.png',height=8,width=10) for i in (5,6)],
              categories=[dict(id=23,name='citizen')],
              annotations=[dict(id=i,image_id=i,category_id=23,mask_file=f'semantic_masks_by_class/{i}.png',area=12) for i in (5,6)])
    path=tmp_path/'UGV-PUI.zip'
    with ZipFile(path,'w') as zipfile:
        zipfile.writestr('UGV-PUI/annotations/semantic_segmentation.json',json.dumps(data))
        for i in (5,6):
            zipfile.writestr(f'UGV-PUI/images/{i}.png',image_bytes)
            zipfile.writestr(f'UGV-PUI/semantic_masks_by_class/{i}.png',mask_bytes)
        zipfile.writestr('UGV-PUI/overlays/unused.jpg',b'not retained')
    return path


def test_import_preserves_masks_deduplicates_images_and_archives_old_bank(tmp_path):
    root=tmp_path/'PUI';root.mkdir();previous=b'old bank retained'
    (root/'bank.json').write_bytes(previous)
    report=module.import_zip(bundle(tmp_path),root)
    assert report['ways']==1 and report['supports']==1
    assert report['one_shot_classes']==['citizen']
    original=root/'history'/('before_'+report['export_sha256'][:12])/'bank.json'
    assert original.read_bytes()==previous
    row=report['classes']['citizen']['examples'][0]
    mask=cv2.imread(str(root/row['mask']),cv2.IMREAD_UNCHANGED)
    assert (mask>0).sum()==12 and mask[0,0]==0
    assert not list(root.rglob('unused.jpg'))


def test_invalid_mask_does_not_replace_active_bank(tmp_path):
    root=tmp_path/'PUI';root.mkdir();(root/'bank.json').write_text('unchanged')
    with pytest.raises(ValueError,match='binary foreground'):
        module.import_zip(bundle(tmp_path,bad=True),root)
    assert (root/'bank.json').read_text()=='unchanged'


def test_uav_import_keeps_uav_task_ids_and_uses_independent_pui_bank(tmp_path):
    source=bundle(tmp_path);target=tmp_path/'UAV-PUI.zip'
    with ZipFile(source) as original,ZipFile(target,'w') as output:
        for name in original.namelist():
            data=original.read(name)
            if name.endswith('semantic_segmentation.json'):
                record=json.loads(data)
                for row in record['images']:row['id']+=24
                for row in record['annotations']:row['image_id']+=24
                data=json.dumps(record).encode()
            output.writestr(name.replace('UGV-PUI/','UAV-PUI/'),data)
    root=tmp_path/'PUI'
    report=module.import_zip(target,root,platform='uav')
    assert report['included_task_ids']==[29,30] and report['excluded_task_ids']==[]
    assert report['source_platform']=='uav' and report['supports']==1
    bank=json.loads((root/'bank.json').read_text())
    assert bank['source_platform']=='uav' and bank['site']=='PUI'
    assert all(not Path(e['image']).is_absolute() for c in report['classes'].values() for e in c['examples'])


def test_raw_import_cannot_overwrite_native_uav_bank(tmp_path):
    root=tmp_path/'PUI';root.mkdir()
    saved=json.dumps(dict(schema_version='triffid.uav.support-bank.v1'))
    (root/'bank.json').write_text(saved)
    with pytest.raises(ValueError,match='Preserve the native UAV bank'):
        module.import_zip(bundle(tmp_path),root,platform='uav')
    assert (root/'bank.json').read_text()==saved
