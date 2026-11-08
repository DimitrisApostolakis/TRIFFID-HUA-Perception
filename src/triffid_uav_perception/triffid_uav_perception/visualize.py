"""Offline UAV detection debugger: JSON + source images -> overlays and gallery.

python -m triffid_uav_perception.visualize point_detections.json --output debug
No model, checkpoints, supports, MP4, SRT or GUI libraries are required.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .export import SCHEMA_VERSION, write_json
from .masks import decode_rle
from .media import read_rgb, write_rgb


def class_color(class_name):
    """RGB, stable across frames, processes and class filters."""
    special = {'civilian vehicle': (40, 210, 255), 'window': (255, 155, 45)}
    if class_name in special:
        return special[class_name]
    digest = hashlib.sha256(class_name.encode('utf-8')).digest()
    return tuple(70 + value % 186 for value in digest[:3])


def _integer_list(values, length):
    return isinstance(values, list) and len(values) == length and all(
        isinstance(v, int) and not isinstance(v, bool) for v in values)


def validate_observation(observation, width, height):
    """Bound mask allocation to its source image and enforce exported coordinates."""
    if not isinstance(observation, dict) or not isinstance(observation.get('class_name'), str):
        raise ValueError('Observation needs a class_name')
    encoded = observation.get('mask')
    if not isinstance(encoded, dict):
        raise ValueError('Observation needs an encoded mask')
    size, offset = encoded.get('size_hw'), encoded.get('offset_xy')
    if not _integer_list(size, 2) or not _integer_list(offset, 2):
        raise ValueError('Invalid mask dimensions/offset')
    h, w = size
    x, y = offset
    if min(h, w) < 1 or min(x, y) < 0 or x + w > width or y + h > height:
        raise ValueError('Mask crop lies outside its source image')
    if observation.get('bbox_xyxy') != [x, y, x + w, y + h]:
        raise ValueError('Bounding box does not match the mask crop')
    centroid = observation.get('centroid_xy')
    if not isinstance(centroid, list) or len(centroid) != 2 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in centroid):
        raise ValueError('Invalid centroid coordinates')
    if not (0 <= centroid[0] < width and 0 <= centroid[1] < height):
        raise ValueError('Centroid lies outside source image')
    crop = decode_rle(encoded)
    if not crop.any():
        raise ValueError('Observation mask is empty')
    return crop, x, y


def _label(image, text, x, y, color, scale, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, image.shape[1] - text_width - 6))
    y = max(text_height + 5, min(y, image.shape[0] - baseline - 3))
    cv2.rectangle(image, (x, y - text_height - 4),
                  (x + text_width + 5, y + baseline + 2), (12, 16, 24), -1)
    cv2.putText(image, text, (x + 2, y - 1), font, scale, color, thickness, cv2.LINE_AA)


def render_observations(rgb, observations, *, alpha=.4, classes=None, min_area_px=1,
                        masks=True, boxes=True, centroids=True, labels=True):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1 or min_area_px < 1:
        raise ValueError('Use alpha in [0,1] and a positive minimum area')
    canvas = rgb.copy()
    height, width = canvas.shape[:2]
    selected = []
    # Tint first, then draw outlines/labels so later masks cannot obscure them.
    for display_id, observation in enumerate(observations, 1):
        if classes is not None and observation.get('class_name', '').lower() not in classes:
            continue
        crop, x, y = validate_observation(observation, width, height)
        area = int(crop.sum())
        if area < min_area_px:
            continue
        color = class_color(observation['class_name'])
        if masks:
            roi = canvas[y:y + crop.shape[0], x:x + crop.shape[1]]
            roi[crop] = np.rint((1 - alpha) * roi[crop] + alpha * np.array(color)).astype(np.uint8)
        selected.append((display_id, observation, color, area))
    thickness = max(1, round(width / 1200))
    scale = max(.4, width / 1920 * .6)
    records = []
    for display_id, observation, color, area in selected:
        x1, y1, x2, y2 = observation['bbox_xyxy']
        if boxes:
            # Exported maximum is exclusive; drawing endpoint is inclusive.
            cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), color, thickness)
        if centroids:
            cx, cy = (int(round(v)) for v in observation['centroid_xy'])
            cv2.drawMarker(canvas, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS,
                           max(9, thickness * 6), thickness + 2)
            cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_CROSS,
                           max(9, thickness * 6), thickness)
        score = observation.get('score')
        if score is not None and (not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise ValueError('Invalid observation score')
        text = f"#{display_id} {observation['class_name']}"
        if score is not None:
            text += f' p={score:.2f}'
        if labels:
            _label(canvas, text, x1, y1 - 8, color, scale, thickness)
        records.append({'display_id': display_id, 'observation_id': observation.get('observation_id'),
                        'class_name': observation['class_name'], 'area_px': area,
                        'bbox_xyxy': observation['bbox_xyxy'], 'centroid_xy': observation['centroid_xy'],
                        'score': score, 'score_type': observation.get('score_type')})
    return canvas, records


def _resize(image, max_width):
    if max_width and image.shape[1] > max_width:
        return cv2.resize(image, (max_width, max(1, round(image.shape[0] * max_width / image.shape[1]))),
                          interpolation=cv2.INTER_AREA)
    return image


def _contact_tile(preview, title, counts):
    tile = np.full((440, 640, 3), (20, 25, 35), np.uint8)
    ratio = min(640 / preview.shape[1], 355 / preview.shape[0])
    scaled = cv2.resize(preview, (max(1, round(preview.shape[1] * ratio)),
                                max(1, round(preview.shape[0] * ratio))), interpolation=cv2.INTER_AREA)
    x = (640 - scaled.shape[1]) // 2
    tile[75:75 + scaled.shape[0], x:x + scaled.shape[1]] = scaled
    cv2.putText(tile, title[:75], (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .58, (245, 245, 245), 1, cv2.LINE_AA)
    text = ', '.join(f'{key}: {value}' for key, value in sorted(counts.items())) or 'No displayed observations'
    cv2.putText(tile, text[:95], (12, 53), cv2.FONT_HERSHEY_SIMPLEX, .44, (200, 210, 220), 1, cv2.LINE_AA)
    return tile


def visualize(json_path, output_dir=None, *, image_root=None, alpha=.4, max_width=1920,
              classes=None, min_area_px=1, masks=True, boxes=True, centroids=True, labels=True):
    path = Path(json_path).resolve()
    output = Path(output_dir).resolve() if output_dir else path.parent / 'debug'
    if max_width < 0:
        raise ValueError('max_width must be nonnegative (0 means original resolution)')
    if min_area_px < 1 or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Use alpha in [0,1] and a positive minimum area')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or data.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'Expected {SCHEMA_VERSION} detection JSON')
    sources = data.get('sources')
    if not isinstance(sources, list) or not sources:
        raise ValueError('Detection JSON contains no source images')
    available = {o['class_name'].lower() for source in sources for o in source.get('observations', [])}
    available.update(c['name'].lower() for c in data.get('class_catalogue', []))
    class_filter = {c.strip().lower() for c in classes.split(',')} if classes else None
    if class_filter is not None and class_filter - available:
        raise ValueError(f'Unknown class filter: {sorted(class_filter - available)}')
    # Check all targets before writing anything, including images referenced by
    # later sources and an input JSON with a reserved debug-output filename.
    targets = {output / name for name in ('contact_sheet.jpg', 'summary.json', 'index.html')}
    targets.update(output / f'{i:02d}_overlay.png' for i in range(len(sources)))
    if path in targets:
        raise ValueError('Debug output would overwrite the input JSON')
    for source in sources:
        reference = source.get('image_file')
        if isinstance(reference, str):
            image = Path(image_root) / Path(reference).name if image_root else path.parent / reference
            if image.resolve() in targets:
                raise ValueError('Debug output would overwrite a source image')
    summaries, tiles = [], []
    for index, source in enumerate(sources):
        reference = source.get('image_file')
        if not isinstance(reference, str) or not reference:
            raise ValueError('Source needs image_file')
        # Explicit relocation override; basename lookup only, never fuzzy/recursive.
        image_path = Path(image_root) / Path(reference).name if image_root else path.parent / reference
        image_path = image_path.resolve()
        # Never overwrite an input image while generating debug output.
        overlay_path = output / f'{index:02d}_overlay.png'
        if image_path == overlay_path:
            raise ValueError('Debug output would overwrite a source image')
        rgb = read_rgb(image_path)
        if rgb.shape[:2] != (source.get('height'), source.get('width')):
            raise ValueError(f'Source dimensions differ from detection JSON: {image_path}')
        observations = source.get('observations', [])
        if not isinstance(observations, list):
            raise ValueError('Source observations must be a list')
        overlay, records = render_observations(rgb, observations, alpha=alpha, classes=class_filter,
            min_area_px=min_area_px, masks=masks, boxes=boxes, centroids=centroids, labels=labels)
        preview = _resize(overlay, max_width)
        write_rgb(overlay_path, preview)
        counts = dict(Counter(record['class_name'] for record in records))
        source_id = str(source.get('source_id', f'source_{index}'))
        title = source_id + (f" | frame {source['frame_index']}" if 'frame_index' in source else '')
        tiles.append(_contact_tile(preview, title, counts))
        summaries.append({'source_id': source_id, 'frame_index': source.get('frame_index'),
                          'image_file': str(image_path), 'overlay_file': overlay_path.name,
                          'input_count': len(observations), 'displayed_count': len(records),
                          'counts': counts, 'observations': records})
    # Four tiles per row: all eight default views fit into two rows.
    columns = min(4, len(tiles))
    sheet = np.full((math.ceil(len(tiles) / columns) * 440, columns * 640, 3), 20, np.uint8)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, columns)
        sheet[row * 440:(row + 1) * 440, col * 640:(col + 1) * 640] = tile
    write_rgb(output / 'contact_sheet.jpg', sheet)
    summary = {'input_json': str(path), 'display_filters': {'classes': sorted(class_filter) if class_filter else None,
               'min_area_px': min_area_px}, 'alpha': alpha, 'max_width': max_width,
               'total_input_observations': sum(s['input_count'] for s in summaries),
               'total_displayed_observations': sum(s['displayed_count'] for s in summaries),
               'sources': summaries}
    write_json(output / 'summary.json', summary)
    cards = []
    for source in summaries:
        title = html.escape(source['source_id'])
        counts = html.escape(json.dumps(source['counts']))
        cards.append(f'<article><h2>{title}</h2><p>{counts}</p>'
                     f'<a href="{source["overlay_file"]}"><img src="{source["overlay_file"]}" alt="{title}"></a></article>')
    (output / 'index.html').write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>UAV detection debugger</title>'
        '<style>body{background:#141923;color:#eee;font:16px system-ui;margin:24px}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:20px}'
        'article{background:#202838;padding:16px;border-radius:8px}img{width:100%;height:auto}'
        'h2{margin:0}p{overflow-wrap:anywhere}a{color:#7dd3fc}</style></head><body>'
        '<h1>UAV detection debugger</h1><p>Masks + boxes + centroid crosses. '
        'Labels # map to display_id in summary.json. p is mean foreground probability, '
        'not calibrated object confidence. Click an image to inspect it.</p>'
        f'<p>{summary["total_displayed_observations"]} displayed / '
        f'{summary["total_input_observations"]} input observations. '
        '<a href="contact_sheet.jpg">Contact sheet</a> · <a href="summary.json">Summary</a></p>'
        '<main>' + ''.join(cards) + '</main></body></html>', encoding='utf-8')
    return output / 'index.html'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json', help='point_detections.json or polygon_segmentations.json')
    parser.add_argument('--output', help='Default: debug/ next to the detection JSON')
    parser.add_argument('--image-root', help='Override source image directory if files have moved')
    parser.add_argument('--alpha', type=float, default=.4)
    parser.add_argument('--max-width', type=int, default=1920, help='Preview width; 0 keeps original resolution')
    parser.add_argument('--classes', help='Comma-separated display class filter')
    parser.add_argument('--min-area-px', type=int, default=1, help='Display-only filter, input JSON unchanged')
    for name in ('masks', 'boxes', 'centroids', 'labels'):
        parser.add_argument('--no-' + name, action='store_true')
    args = parser.parse_args(argv)
    try:
        result = visualize(args.json, args.output, image_root=args.image_root, alpha=args.alpha,
                           max_width=args.max_width, classes=args.classes, min_area_px=args.min_area_px,
                           masks=not args.no_masks, boxes=not args.no_boxes,
                           centroids=not args.no_centroids, labels=not args.no_labels)
    except (ValueError, OSError, KeyError, TypeError, cv2.error) as error:
        parser.exit(2, f'UAV visualization error: {error}\n')
    print(result)


if __name__ == '__main__':
    main()
