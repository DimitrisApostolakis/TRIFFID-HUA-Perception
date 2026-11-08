"""Export UAV-PUI without changing Label Studio data.

Run with an environment containing numpy, Pillow, and pycocotools.
"""
import colorsys
import argparse
import hashlib
import json
import math
import shutil
import sqlite3
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha256(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def decode_brush(values, width, height):
    """Decode the Label Studio MSB-first run-length bit stream, taking alpha.

    Format reference: HumanSignal/label-studio-converter, brush.py.
    """
    source = bytes(values)
    bitpos = 0

    def read(n):
        nonlocal bitpos
        assert bitpos + n <= len(source) * 8, 'Truncated brush RLE'
        start = bitpos // 8
        end = (bitpos + n + 7) // 8
        shift = end * 8 - bitpos - n
        value = (int.from_bytes(source[start:end], 'big') >> shift) & ((1 << n) - 1)
        bitpos += n
        return value

    length = read(32)
    assert length == width * height * 4, (length, width, height)
    word = read(5) + 1
    assert word == 8
    sizes = [read(4) + 1 for _ in range(4)]
    output = np.zeros(length, dtype=np.uint8)
    pos = 0
    while pos < length:
        repeated = read(1)
        end = pos + 1 + read(sizes[read(2)])
        assert end <= length, 'Brush RLE run exceeds image'
        if repeated:
            output[pos:end] = read(word)
        else:
            output[pos:end] = [read(word) for _ in range(end - pos)]
        pos = end
    return output.reshape(height, width, 4)[:, :, 3] > 0


def rle(mask):
    value = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    value['counts'] = value['counts'].decode('ascii')
    assert np.array_equal(coco_mask.decode(value).astype(bool), mask)
    return value


def bbox(mask):
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def box_corners(result, width, height):
    v = result['value']
    x, y = v['x'] * width / 100, v['y'] * height / 100
    w, h = v['width'] * width / 100, v['height'] * height / 100
    a = math.radians(v.get('rotation', 0))
    return [(x + dx * math.cos(a) - dy * math.sin(a), y + dx * math.sin(a) + dy * math.cos(a))
            for dx, dy in [(0, 0), (w, 0), (w, h), (0, h)]]


def draw_boxes(image, boxes, palette, font):
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for box in boxes:
        color = tuple(palette[box['category_id']])
        points = [tuple(x) for x in box['rotated_corners']]
        draw.line(points + [points[0]], fill=color, width=3)
        x = max(0, min(image.width - 1, points[0][0]))
        y = max(0, min(image.height - 24, points[0][1] - 22))
        text = box['category_name']
        bounds = draw.textbbox((x, y), text, font=font)
        draw.rectangle(bounds, fill=color)
        draw.text((x, y), text, fill='black', font=font)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='UAV-PUI', help='Exact saved Label Studio project title')
    parser.add_argument('--storage',type=Path,required=True,help='Read-only Label Studio data directory')
    parser.add_argument('--output',type=Path,required=True,help='New unpacked export directory')
    parser.add_argument('--archive',type=Path,required=True,help='New portable ZIP destination')
    args=parser.parse_args()
    now = datetime.now(timezone.utc)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    for folder in ['images', 'annotations', 'semantic_masks', 'semantic_masks_by_class',
                   'overlays/object_detection', 'overlays/semantic', 'overlays/combined', 'metadata']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    # Read one consistent snapshot. No database or project settings are modified.
    db = sqlite3.connect('file:' + str(args.storage / 'label_studio.sqlite3') + '?mode=ro', uri=True)
    snapshot = sqlite3.connect(':memory:')
    snapshot.row_factory = sqlite3.Row
    db.backup(snapshot)
    db.close()
    row = snapshot.execute('SELECT * FROM project WHERE title=? AND deleted_at IS NULL', (args.project,)).fetchone()
    if row is None:
        raise ValueError(f'Project not found: {args.project}')
    project = dict(row)
    tasks = [dict(r) for r in snapshot.execute('SELECT * FROM task WHERE project_id=? ORDER BY id', (project['id'],))]
    xml = project['label_config']
    (out / 'metadata/label_config.xml').write_text(xml)
    cats = []
    for label in ET.fromstring(xml).find('RectangleLabels').findall('Label'):
        cid = int(label.attrib['category'])
        color = [int(v * 255) for v in colorsys.hsv_to_rgb((cid * 0.61803398875) % 1, 0.65, 1)]
        cats.append(dict(id=cid, name=label.attrib['value'], supercategory=args.project, color=color))
    by_name = {c['name']: c['id'] for c in cats}
    palette = np.zeros((65536, 3), dtype=np.uint8)
    for category in cats:
        palette[category['id']] = category['color']
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    info = dict(description=args.project+' saved annotations; no panoptic export',
                date_created=now.isoformat(), source_project_id=project['id'], source_project=args.project)
    detection = dict(info=info, images=[], annotations=[], categories=cats)
    semantic = dict(info=info, format='COCO-style semantic segmentation', images=[], annotations=[], categories=cats,
                    mask_encoding=dict(png_dtype='uint16', pixel_value='category_id', unlabeled_value=65535),
                    overlap_policy='Per-class masks preserve overlaps; single-label maps give smaller masks priority, then later result order.')
    raw = []
    manifest = []
    all_regions = []
    totals = Counter()
    warnings = []
    det_id = 0
    sem_id = 0
    for task in tasks:
        task_id = task['id']
        data = json.loads(task['data'])
        image_url = data.get('image', data.get('$undefined$'))
        assert image_url and image_url.startswith('/data/upload/'), (task_id, data)
        source = args.storage / 'media' / image_url.removeprefix('/data/')
        original_name = source.name.split('-', 1)[-1]
        image_name = f'task_{task_id:06d}_{original_name}'
        image_rel = 'images/' + image_name
        shutil.copy2(source, out / image_rel)
        assert sha256(source) == sha256(out / image_rel)
        with Image.open(source) as im:
            image = im.convert('RGB')
        width, height = image.size
        records = [dict(r) for r in snapshot.execute('SELECT * FROM task_completion WHERE task_id=? ORDER BY updated_at,id', (task_id,))]
        valid = [r for r in records if not r['was_cancelled'] and r['result']]
        selected = valid[-1] if valid else None
        if len(valid) > 1:
            warnings.append(f'Task {task_id}: selected latest saved annotation {selected["id"]}; all annotations included in native JSON.')
        results = json.loads(selected['result']) if selected else []
        native_annotations = []
        for row in records:
            native_annotations.append({k: json.loads(v) if k in ('result', 'prediction') and v else v for k, v in row.items()})
        raw.append(dict(id=task_id, data=dict(image=image_rel), source_data=data,
                        annotations=native_annotations, meta=json.loads(task['meta'] or '{}'), project=project['id']))
        image_record = dict(id=task_id, file_name=image_rel, width=width, height=height,
                            label_studio_task_id=task_id, label_studio_annotation_id=selected['id'] if selected else None)
        detection['images'].append(image_record)
        sem_image = dict(image_record, semantic_mask_file=f'semantic_masks/task_{task_id:06d}.png')
        semantic['images'].append(sem_image)
        boxes = []
        brush = []
        per_class = {}
        for index, result in enumerate(results):
            kind = result.get('type')
            if kind not in ('brushlabels', 'rectanglelabels'):
                continue
            assert result.get('image_rotation', 0) == 0, 'Rotated image needs coordinate conversion'
            assert (result['original_width'], result['original_height']) == (width, height), (task_id, image.size)
            names = result['value'][kind]
            assert len(names) == 1, (task_id, names)
            category_id = by_name[names[0]]
            if kind == 'rectanglelabels':
                recovered = not all(k in result['value'] for k in ['x', 'y', 'width', 'height'])
                if recovered:
                    assert result['value'].get('format') == 'rle'
                    recovered_mask = decode_brush(result['value']['rle'], width, height)
                    assert recovered_mask.any()
                    mx, my, mw, mh = bbox(recovered_mask)
                    corners = [(mx, my), (mx+mw, my), (mx+mw, my+mh), (mx, my+mh)]
                    totals['boxes_recovered_from_stored_rle'] += 1
                    warnings.append(f'Task {task_id}: box {result["id"]} has RLE instead of coordinates; exported its mask bounding box.')
                    # Avoid counting duplicated mask/box results twice in the semantic export.
                    matching_brush = [r for r in results if r.get('type') == 'brushlabels'
                                      and r.get('id') == result['id']
                                      and r['value'].get('brushlabels') == names
                                      and r['value'].get('rle') == result['value']['rle']]
                    if not matching_brush:
                        brush.append((recovered_mask, category_id, int(recovered_mask.sum()), index))
                        if category_id not in per_class:
                            per_class[category_id] = recovered_mask.copy()
                        else:
                            per_class[category_id] |= recovered_mask
                        all_regions.append(dict(task_id=task_id, annotation_id=selected['id'], region_id=result['id'],
                                                category_id=category_id, category_name=names[0],
                                                area=int(recovered_mask.sum()), source_type='rectanglelabels_with_rle'))
                        totals['extra_masks_recovered_from_rectangle_rle'] += 1
                else:
                    corners = box_corners(result, width, height)
                x0, y0 = np.min(corners, axis=0)
                x1, y1 = np.max(corners, axis=0)
                x0, x1 = max(0., float(x0)), min(float(width), float(x1))
                y0, y1 = max(0., float(y0)), min(float(height), float(y1))
                assert x1 > x0 and y1 > y0
                det_id += 1
                entry = dict(id=det_id, image_id=task_id, category_id=category_id, category_name=names[0],
                             bbox=[x0, y0, x1-x0, y1-y0], area=(x1-x0)*(y1-y0), iscrowd=0,
                             rotated_corners=corners, rotation=result['value'].get('rotation', 0),
                             source_region_id=result['id'], source_annotation_id=selected['id'],
                             geometry_source='stored_mask_bounding_box' if recovered else 'rectangle_coordinates')
                boxes.append(entry)
                detection['annotations'].append(entry)
                totals['boxes'] += 1
            else:
                totals['brush_records'] += 1
                if result['value'].get('format') != 'rle':
                    totals['brush_records_without_mask_pixels'] += 1
                    warnings.append(f'Task {task_id}: brush {result["id"]} ({names[0]}) contains rectangle coordinates, not mask pixels; retained in native JSON but excluded from semantic targets.')
                    continue
                mask = decode_brush(result['value']['rle'], width, height)
                area = int(mask.sum())
                totals['brush_regions'] += 1
                if not area:
                    warnings.append(f'Task {task_id}: brush region {result["id"]} is empty.')
                brush.append((mask, category_id, area, index))
                if category_id not in per_class:
                    per_class[category_id] = mask.copy()
                else:
                    per_class[category_id] |= mask
                all_regions.append(dict(task_id=task_id, annotation_id=selected['id'], region_id=result['id'],
                                        category_id=category_id, category_name=names[0], area=area))
        # Lossless class union masks retain cross-class overlaps; these are the semantic JSON targets.
        for category_id, mask in sorted(per_class.items()):
            if not mask.any():
                continue
            sem_id += 1
            rel = f'semantic_masks_by_class/task_{task_id:06d}_class_{category_id:02d}.png'
            Image.fromarray(mask.astype(np.uint8) * 255).save(out / rel)
            segmentation = rle(mask)
            semantic['annotations'].append(dict(id=sem_id, image_id=task_id, category_id=category_id,
                                               segmentation=segmentation, area=int(mask.sum()), bbox=bbox(mask),
                                               mask_file=rel, iscrowd=0))
        single = np.full((height, width), 65535, dtype=np.uint16)
        coverage = np.zeros((height, width), dtype=np.uint16)
        for mask, category_id, area, index in sorted(brush, key=lambda b: (-b[2], b[3])):
            single[mask] = category_id
        for mask in per_class.values():
            coverage += mask
        totals['cross_class_overlap_pixels'] += int((coverage > 1).sum())
        Image.fromarray(single).save(out / sem_image['semantic_mask_file'])
        colored = palette[single]
        visible = single != 65535
        pixels = np.array(image)
        pixels[visible] = (pixels[visible].astype(float) * 0.55 + colored[visible].astype(float) * 0.45).astype(np.uint8)
        overlay = Image.fromarray(pixels)
        draw_boxes(image, boxes, palette, font).save(out / f'overlays/object_detection/task_{task_id:06d}.jpg', quality=94)
        overlay.save(out / f'overlays/semantic/task_{task_id:06d}.jpg', quality=94)
        draw_boxes(overlay, boxes, palette, font).save(out / f'overlays/combined/task_{task_id:06d}.jpg', quality=94)
        # Read back maps and overlays to verify dimensions and pixel-level correspondence.
        with Image.open(out / sem_image['semantic_mask_file']) as im:
            assert np.array_equal(np.asarray(im), single)
        for folder in ['object_detection', 'semantic', 'combined']:
            with Image.open(out / f'overlays/{folder}/task_{task_id:06d}.jpg') as im:
                assert im.size == image.size
                im.verify()
        for category_id, mask in per_class.items():
            if not mask.any():
                continue
            with Image.open(out / f'semantic_masks_by_class/task_{task_id:06d}_class_{category_id:02d}.png') as im:
                assert np.array_equal(np.asarray(im) > 0, mask)
        manifest.append(dict(task_id=task_id, annotation_id=selected['id'] if selected else None,
                             source_image_url=image_url, image_file=image_rel, source_image_sha256=sha256(source),
                             boxes=len(boxes), brush_regions=len(brush), semantic_classes=len(per_class)))
        totals['images'] += 1
        totals['saved_annotations'] += len(valid)
        print(f'Exported {totals["images"]}/{len(tasks)}: task {task_id}, {len(boxes)} boxes, {len(brush)} masks', flush=True)
    dump(out / 'annotations/object_detection.json', detection)
    dump(out / 'annotations/semantic_segmentation.json', semantic)
    dump(out / 'annotations/label_studio.json', raw)
    dump(out / 'metadata/categories.json', cats)
    dump(out / 'metadata/manifest.json', manifest)
    dump(out / 'metadata/brush_regions.json', all_regions)
    dump(out / 'metadata/export_report.json', dict(project=args.project, created_at=now.isoformat(), totals=totals, warnings=warnings))
    legend = Image.new('RGB', (1200, math.ceil(len(cats)/3)*36+45), 'white')
    draw = ImageDraw.Draw(legend)
    draw.text((10, 8), args.project + ' category IDs and overlay colours', fill='black', font=font)
    for i, category in enumerate(cats):
        x, y = (i % 3)*400+10, (i//3)*36+40
        draw.rectangle((x, y, x+24, y+24), fill=tuple(category['color']))
        draw.text((x+32, y), f'{category["id"]}: {category["name"]}', fill='black', font=font)
    legend.save(out / 'metadata/category_legend.png')
    readme = '''# UAV-PUI export

This package contains only project UAV-PUI (Label Studio project 6), from a consistent database snapshot.
No panoptic files are included, as requested. Source images are byte-for-byte copies.

## Files

- images/: original images, named with stable Label Studio task IDs.
- annotations/object_detection.json: COCO object detection JSON. bbox is [x,y,width,height] in pixels.
  Rotated boxes use their clipped axis-aligned envelope for bbox; original rotated corners and angle are included.
- annotations/semantic_segmentation.json: COCO-style semantic JSON, one merged mask per image/category,
  with compressed COCO RLE, area, bounding box and corresponding binary PNG path.
- semantic_masks_by_class/: lossless binary PNGs (0 background, 255 annotated class). Class masks may overlap.
- semantic_masks/: single-label 16-bit PNGs, pixel value = original category ID (0..62).
  65535 means UNANNOTATED/IGNORE, not background. Never interpret unlabeled pixels as negative examples.
  Smaller source brush regions take priority in overlaps; ties go to later result order.
  Use per-class PNGs/JSON RLE or native annotations to retain all overlapping labels.
- overlays/semantic/: class-coloured mask overlays. These visualizations use the single-label maps.
- overlays/object_detection/: labelled box overlays.
- overlays/combined/: masks and boxes together. All overlays are at original image resolution, JPEG quality 94.
- annotations/label_studio.json: saved native annotation results, including original brush RLE and prompts.
  data.image is portable within this package; source_data retains the original Label Studio URL.
- metadata/: XML configuration, class IDs/colour legend, image/task manifest, brush region metadata,
  export counts, checksums and any conversion warnings.
- export_uav_pui.py: reproducible export script (requires original database/media, numpy, Pillow, pycocotools).

## Selection and provenance

Only saved annotations are used; drafts and predictions are not substituted for annotations.
If several saved non-cancelled annotations exist, derived files use the most recently updated annotation;
all saved native annotations are retained in label_studio.json. No class labels or masks are inferred by a model.
Semantic masks come exclusively from brush regions. Boxes without a brush mask remain detection-only.
Exception: malformed rectangle records carrying stored mask RLE are recovered from that RLE. Their
boxes are marked geometry_source=stored_mask_bounding_box. If no identical brush record exists,
the stored mask also contributes to semantic targets. Each recovery is logged; native records are unchanged.
Brush records containing rectangle coordinates instead of mask pixels cannot supply semantic masks;
they are retained in native JSON and listed in export_report.json, and no filled rectangle is invented.
Category IDs preserve the XML category values; category 0 is water, not background.
All images are retained even if an image has no box or mask.
Overlay colours are deterministic export colours, not necessarily the Label Studio display palette.
Mask/box instances are not automatically linked; original region IDs remain in the metadata/native JSON.

## Format references

- Label Studio brush RLE: https://github.com/HumanSignal/label-studio-converter/blob/master/label_studio_converter/brush.py
- COCO detection/RLE: https://cocodataset.org/#format-data
'''
    readme = readme.replace('UAV-PUI', args.project).replace('project 6', f'project {project["id"]}')
    (out / 'README.md').write_text(readme)
    shutil.copy2(Path(__file__), out / 'export_uav_pui.py')
    # Load both indexes with the COCO API and check the full task population.
    for name in ['object_detection.json', 'semantic_segmentation.json']:
        coco = COCO(str(out / 'annotations' / name))
        assert set(coco.getImgIds()) == {t['id'] for t in tasks}
    checksums = {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    dump(out / 'metadata/sha256.json', checksums)
    zip_path = args.archive
    zip_path.parent.mkdir(parents=True,exist_ok=True)
    assert not zip_path.exists(), 'Refusing to overwrite an existing ZIP'
    temporary_zip = zip_path.with_name(zip_path.name+'.partial')
    with zipfile.ZipFile(temporary_zip, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                archive.write(path, args.project+'/' + path.relative_to(out).as_posix())
    with zipfile.ZipFile(temporary_zip) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(checksums) + 1
    temporary_zip.rename(zip_path)
    print(json.dumps(dict(zip=str(zip_path), bytes=zip_path.stat().st_size, sha256=sha256(zip_path),
                          files=len(checksums)+1, totals=totals, warnings=warnings), indent=2), flush=True)


if __name__ == '__main__':
    main()
