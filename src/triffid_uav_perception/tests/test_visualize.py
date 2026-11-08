import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from triffid_uav_perception.export import document, write_json
from triffid_uav_perception.masks import observations_from_prediction
from triffid_uav_perception.media import write_rgb
from triffid_uav_perception.models.base import ClassPrediction
from triffid_uav_perception.ontology import BY_ID, classes_for
from triffid_uav_perception.visualize import class_color, render_observations, visualize


def observation(class_id=21):
    mask = np.zeros((120, 180), bool)
    mask[30:90, 40:140] = True
    mask[45:75, 65:115] = False
    return observations_from_prediction(ClassPrediction(class_id, mask), BY_ID[class_id], 'view_00')[0]


def test_crop_offsets_holes_and_mask_alpha():
    original = np.zeros((120, 180, 3), np.uint8)
    rendered, records = render_observations(original, [observation()], alpha=.5,
                                           boxes=False, centroids=False, labels=False)
    assert np.array_equal(rendered[35, 45], np.rint(np.array(class_color('civilian vehicle')) * .5))
    assert not rendered[60, 90].any()  # Hole remains background.
    assert not rendered[20, 20].any()
    assert not original.any()  # Source image not modified in-place.
    assert records[0]['area_px'] == 4500
    assert records[0]['display_id'] == 1


def test_filtering_is_display_only_and_ids_remain_traceable():
    observations = [observation(), observation(13)]
    before = json.dumps(observations)
    _, records = render_observations(np.zeros((120, 180, 3), np.uint8), observations,
                                     classes={'window'}, labels=False)
    assert records[0]['class_name'] == 'window' and records[0]['display_id'] == 2
    assert json.dumps(observations) == before
    _, records = render_observations(np.zeros((120, 180, 3), np.uint8), observations, min_area_px=5000)
    assert not records


@pytest.mark.parametrize('mutation', ['bounds', 'rle', 'bbox', 'centroid'])
def test_invalid_mask_geometry_rejected(mutation):
    item = observation()
    if mutation == 'bounds':
        item['mask']['offset_xy'] = [10000, 10000]
    elif mutation == 'rle':
        item['mask']['counts'] = [1000000]
    elif mutation == 'bbox':
        item['bbox_xyxy'][2] += 1
    else:
        item['centroid_xy'] = [float('nan'), 0]
    with pytest.raises(ValueError):
        render_observations(np.zeros((120, 180, 3), np.uint8), [item])


def fixture_document(tmp_path, absolute=False, count=8):
    image_root = tmp_path / 'frames'
    sources = []
    for i in range(count):
        path = image_root / f'view_{i:02d}.png'
        write_rgb(path, np.full((120, 180, 3), 30, np.uint8))
        sources.append({'source_id': f'view_{i:02d}', 'frame_index': i * 10,
                        'width': 180, 'height': 120,
                        'image_file': str(path) if absolute else f'frames/{path.name}',
                        'observations': [observation(), observation(13)]})
    payload = document('point_detections', sources, {}, classes_for('Point', 'civilian vehicle,window'))
    return write_json(tmp_path / 'detections.json', payload)


@pytest.mark.parametrize('absolute', [True, False])
def test_eight_frame_gallery_end_to_end(tmp_path, absolute):
    path = fixture_document(tmp_path, absolute)
    before = path.read_bytes()
    image_hashes = [p.read_bytes() for p in sorted((tmp_path / 'frames').glob('*.png'))]
    gallery = visualize(path, max_width=90)
    assert gallery.is_file()
    assert len(list(gallery.parent.glob('*_overlay.png'))) == 8
    assert cv2.imread(str(gallery.parent / '00_overlay.png')).shape == (60, 90, 3)
    assert cv2.imread(str(gallery.parent / 'contact_sheet.jpg')).shape == (880, 2560, 3)
    summary = json.loads((gallery.parent / 'summary.json').read_text())
    assert summary['total_input_observations'] == summary['total_displayed_observations'] == 16
    assert summary['sources'][0]['counts'] == {'civilian vehicle': 1, 'window': 1}
    assert path.read_bytes() == before
    assert [p.read_bytes() for p in sorted((tmp_path / 'frames').glob('*.png'))] == image_hashes


def test_relocated_images_class_filter_and_original_resolution(tmp_path):
    path = fixture_document(tmp_path, True)
    root = tmp_path / 'frames'
    moved = tmp_path / 'moved'
    root.rename(moved)
    gallery = visualize(path, image_root=moved, classes='window', max_width=0)
    assert cv2.imread(str(gallery.parent / '00_overlay.png')).shape == (120, 180, 3)
    summary = json.loads((gallery.parent / 'summary.json').read_text())
    assert summary['total_displayed_observations'] == 8
    with pytest.raises(ValueError, match='Unknown class'):
        visualize(path, image_root=moved, classes='nonexistent')


def test_no_detections_is_valid_and_unsafe_html_is_escaped(tmp_path):
    path = fixture_document(tmp_path, count=1)
    data = json.loads(path.read_text())
    data['sources'][0]['observations'] = []
    data['sources'][0]['source_id'] = '<script>alert(1)</script>'
    path.write_text(json.dumps(data))
    gallery = visualize(path)
    assert '<script>' not in gallery.read_text()
    assert '&lt;script&gt;' in gallery.read_text()
    assert json.loads((gallery.parent / 'summary.json').read_text())['total_displayed_observations'] == 0


def test_input_json_and_source_images_cannot_be_overwritten(tmp_path):
    path = fixture_document(tmp_path, True, count=1)
    summary = tmp_path / 'summary.json'
    summary.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match='input JSON'):
        visualize(summary, tmp_path)
    data = json.loads(path.read_text())
    source = tmp_path / '00_overlay.png'
    write_rgb(source, np.zeros((120, 180, 3), np.uint8))
    data['sources'][0]['image_file'] = str(source)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='source image'):
        visualize(path, tmp_path)
