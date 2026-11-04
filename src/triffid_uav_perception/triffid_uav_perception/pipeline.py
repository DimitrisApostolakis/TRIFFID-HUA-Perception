"""Orchestrates IO, angular selection, a replaceable backend and JSON export."""
from pathlib import Path
import json
import logging

from .export import document, write_json
from .frame_selection import select_views
from .masks import observations_from_prediction
from .media import VideoReader, read_rgb, write_rgb
from .srt_metadata import find_sidecar_srt, parse_srt

log = logging.getLogger(__name__)


def prepare_views(video, srt, selection):
    srt_path = Path(srt) if srt else find_sidecar_srt(video)
    if srt_path is None or not srt_path.is_file():
        raise ValueError('A matching DJI SRT is required for angular frame selection')
    records = parse_srt(srt_path.read_text(encoding='utf-8-sig', errors='replace'))
    with VideoReader(video) as reader:
        views, settings = select_views(records, reader.info.fps, reader.info.frame_count, **selection)
        info = reader.info
    return info, srt_path.resolve(), views, settings


def source_from_view(view, rgb, image_path):
    source = view.to_dict()
    source.update({'source_id': view.view_id, 'source_type': 'video_frame',
                   'image_file': image_path, 'width': rgb.shape[1], 'height': rgb.shape[0],
                   'telemetry_time_delta_s': view.timestamp_s - view.telemetry.timestamp_s,
                   'observations': []})
    return source


def select_video_frames(video, srt, output_dir, **selection):
    info, srt_path, views, settings = prepare_views(video, srt, selection)
    output_dir = Path(output_dir)
    sources = []
    with VideoReader(video) as reader:
        for view in views:
            rgb = reader.read_rgb(view.frame_index)
            relative = f'frames/{view.view_id}.png'
            write_rgb(output_dir / relative, rgb)
            sources.append(source_from_view(view, rgb, relative))
    return write_json(output_dir / 'selected_frames.json', {
        'schema_version': 'triffid.uav.selection.v1', 'video': info.to_dict(),
        'srt_file': str(srt_path), 'selection': settings, 'sources': sources})


def infer_source(backend, rgb, classes, source, min_area_px):
    expected = {c.id: c for c in classes}
    seen = set()
    for prediction in backend.predict(rgb, classes):
        if prediction.class_id not in expected or prediction.class_id in seen:
            raise ValueError('Backend returned an unexpected or repeated class')
        seen.add(prediction.class_id)
        if prediction.mask.shape != rgb.shape[:2]:
            raise ValueError('Backend mask dimensions must match the original input image')
        source['observations'].extend(observations_from_prediction(
            prediction, expected[prediction.class_id], source['source_id'], min_area_px=min_area_px))
    if seen != set(expected):
        raise ValueError('Backend must return a prediction per requested class, including empty masks')
    log.info('%s: %s observations', source['source_id'], len(source['observations']))


def detect_video(video, srt, output_dir, backend, classes, *, min_area_px=1, **selection):
    info, srt_path, views, settings = prepare_views(video, srt, selection)
    backend.prepare(classes)
    sources = []
    output_dir = Path(output_dir)
    with VideoReader(video) as reader:
        # Decode/infer one selected frame at a time; no full-video RAM/GPU buffer.
        for view in views:
            rgb = reader.read_rgb(view.frame_index)
            relative = f'frames/{view.view_id}.png'
            write_rgb(output_dir / relative, rgb)
            source = source_from_view(view, rgb, relative)
            infer_source(backend, rgb, classes, source, min_area_px)
            sources.append(source)
    payload = document('point_detections', sources, backend.metadata(), classes)
    payload.update({'video': info.to_dict(), 'srt_file': str(srt_path), 'selection': settings,
                    'postprocessing': {'min_area_px': min_area_px, 'connectivity': 8}})
    return write_json(output_dir / 'point_detections.json', payload)


def segment_top_view(image, output_dir, backend, classes, *, projection_json=None, min_area_px=1):
    rgb = read_rgb(image)
    projection = None
    if projection_json:
        projection = json.loads(Path(projection_json).read_text())
        if not isinstance(projection, dict):
            raise ValueError('Top-view projection context must be a JSON object')
    backend.prepare(classes)
    source = {'source_id': 'top_view', 'source_type': 'top_view_image',
              'image_file': str(Path(image).resolve()), 'width': rgb.shape[1], 'height': rgb.shape[0],
              'projection_context': projection, 'observations': []}
    infer_source(backend, rgb, classes, source, min_area_px)
    payload = document('polygon_segmentation', [source], backend.metadata(), classes)
    payload['postprocessing'] = {'min_area_px': min_area_px, 'connectivity': 8}
    return write_json(Path(output_dir) / 'polygon_segmentations.json', payload)
