"""Regression checks for leakage exclusion and the fixed-slot support contract."""
import importlib.util
import json
from pathlib import Path
from PIL import Image, PngImagePlugin
import numpy as np
from triffid_ugv_perception.supports import SupportBank
from triffid_ugv_perception.ontology import classes_for

SCRIPT=Path(__file__).resolve().parents[3]/'scripts/prepare_rosbag_pui_eval.py'
spec=importlib.util.spec_from_file_location('prepare_rosbag_pui_eval',SCRIPT)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_pixel_identical_query_removed_and_reserved_slots_remain_valid(tmp_path):
    query=tmp_path/'query.png'
    Image.new('RGB',(8,8),(50,100,150)).save(query)
    duplicate=tmp_path/'reencoded.png'
    metadata=PngImagePlugin.PngInfo();metadata.add_text('encoding','different bytes, same image')
    Image.open(query).save(duplicate,pnginfo=metadata)
    other=tmp_path/'independent.png'
    Image.new('RGB',(8,8),(150,100,50)).save(other)
    mask=tmp_path/'mask.png'
    Image.fromarray(np.full((8,8),255,np.uint8)).save(mask)
    source=tmp_path/'source.json'
    source.write_text(json.dumps(dict(schema_version='triffid.ugv.support-bank.v1',examples_per_class=16,
        classes={'fence':[dict(id='query_copy',image=str(duplicate),mask=str(mask)),
                          dict(id='support',image=str(other),mask=str(mask))]})))
    before=source.read_bytes()
    assert module.digest(query)!=module.digest(duplicate)
    bank,removed,counts=module.freeze_bank(source,{module.pixels(query)})
    assert source.read_bytes()==before
    assert len(removed)==1 and counts=={'fence':1}
    assert len(bank['classes']['fence'])==16
    target=tmp_path/'frozen.json';target.write_text(json.dumps(bank))
    selected=SupportBank(target).select_available(classes_for('fence'))
    supports=selected.for_class(classes_for('fence')[0])
    assert len(supports)==1 and supports[0].image==other

def test_fully_held_out_class_is_explicitly_unavailable(tmp_path):
    image=tmp_path/'query.png';mask=tmp_path/'mask.png'
    Image.new('RGB',(8,8),'red').save(image)
    Image.new('L',(8,8),255).save(mask)
    source=tmp_path/'source.json'
    source.write_text(json.dumps(dict(schema_version='triffid.ugv.support-bank.v1',examples_per_class=16,
        classes={'fence':[dict(image=str(image),mask=str(mask))]})))
    bank,removed,counts=module.freeze_bank(source,{module.pixels(image)})
    assert counts=={} and len(removed)==1
    assert not Path(bank['classes']['fence'][0]['image']).exists()
