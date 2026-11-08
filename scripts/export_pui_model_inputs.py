"""Prepare annotation-only inputs for all backends; no feature extraction."""
import json
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_utils
from triffid_ugv_perception.supports import SupportBank
from triffid_ugv_perception.ontology import classes_for

ROOT=Path('/mnt/d/projects/triffid-hua-perception')


def main():
    bank_root=ROOT/'ugv_supports/PUI'
    report=json.loads((bank_root/'import_report.json').read_text())
    classes=classes_for(','.join(report['one_shot_classes']))
    selected=SupportBank(bank_root).select_available(classes)
    manifest=dict(schema='triffid.model_supports.v1',bank='ugv_supports/PUI',
        bank_sha256=selected.manifest_sha256,ways=len(classes),supports=0,classes=[])
    native_fsod={}
    coco=dict(info=dict(description='PUI UGV visual supports'),licenses=[],images=[],annotations=[],categories=[])
    image_ids={}
    for index,item in enumerate(classes):
        examples=[]
        coco['categories'].append(dict(id=index+1,name=item.name,supercategory='PUI',isthing=1))
        native_fsod[item.name]=[]
        for support in selected.for_class(item):
            rgb,mask=support.load()
            ys,xs=np.where(mask)
            bbox=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
            xywh=[bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]]
            image=str(support.image.relative_to(bank_root))
            mask_path=str(support.mask.relative_to(bank_root))
            examples.append(dict(example_id=support.example_id,image=image,mask=mask_path,
                foreground_value=support.foreground_value,bbox_xyxy=bbox,width=rgb.shape[1],height=rgb.shape[0]))
            native_fsod[item.name].append(dict(image=image,bbox=xywh,annotated_mask=mask_path))
            if image not in image_ids:
                image_ids[image]=len(image_ids)+1
                coco['images'].append(dict(id=image_ids[image],file_name=image,width=rgb.shape[1],height=rgb.shape[0]))
            rle=mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts']=rle['counts'].decode('ascii')
            coco['annotations'].append(dict(id=len(coco['annotations'])+1,image_id=image_ids[image],
                category_id=index+1,bbox=xywh,area=int(mask.sum()),iscrowd=0,segmentation=rle))
        manifest['classes'].append(dict(index=index,ontology_id=item.id,name=item.name,examples=examples))
        manifest['supports']+=len(examples)
    destination=bank_root/'model_inputs'
    destination.mkdir(exist_ok=True)
    for name,data in [('manifest.json',manifest),('fsod_vfm_supports.json',native_fsod),('supports_coco.json',coco)]:
        (destination/name).write_text(json.dumps(data,indent=2)+'\n')
    print(f'Annotation inputs prepared: {manifest["ways"]} classes, {manifest["supports"]} supports; no model run',flush=True)


if __name__=='__main__':main()
