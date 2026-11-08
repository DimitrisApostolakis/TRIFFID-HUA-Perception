import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from triffid_uav_perception.ontology import BY_ID, CLASSES, classes_for
from triffid_uav_perception.supports import (
    BANK_SCHEMA, SupportBank, add_example, bank_template, initialize_bank,
)
from triffid_uav_perception.uav_node import build_parser, main


def test_initialize_all_classes_with_5_empty_slots_without_fake_images(tmp_path):
    root = tmp_path / 'bank'
    manifest = initialize_bank(root)
    bank = SupportBank(root)
    assert json.loads(manifest.read_text()) == bank_template()
    assert len(bank.data['classes']) == 38
    assert sum(len(bank.for_class(c)) for c in CLASSES) == 190
    assert not list(root.rglob('*.png'))
    report = bank.validate()
    assert not report['valid'] and report['expected_examples'] == 190
    assert report['valid_examples'] == 0 and len(report['errors']) == 190
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match='already exists'):
        initialize_bank(root)
    assert manifest.read_bytes() == before


def test_deterministic_five_and_one_shot_selection(bank_selection):
    classes = classes_for('Point', 'debris,citizen')
    selection = bank_selection(classes)
    bank = SupportBank(Path(selection.examples[0][1][0].image).parent)
    one = bank.select(classes, shots=1)
    again = bank.select(classes, shots=5)
    assert selection == again
    for c in classes:
        assert one.for_class(c) == selection.for_class(c)[:1]
        assert len(selection.for_class(c)) == 5
    assert one.metadata()['shots'] == 1
    assert len(one.metadata()['classes']['citizen']) == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        selection.shots = 1


def test_different_model_backends_receive_identical_examples(bank_selection):
    classes = classes_for('Point', 'citizen')
    selection = bank_selection(classes)
    class FirstModel:
        def prepare(self, classes, supports):
            self.examples = tuple(supports.for_class(c) for c in classes)
    class SecondModel:
        def prepare(self, classes, supports):
            self.examples = tuple(supports.for_class(c) for c in classes)
    a, b = FirstModel(), SecondModel()
    a.prepare(classes, selection)
    b.prepare(classes, selection)
    assert a.examples == b.examples


def test_provenance_changes_when_bank_assets_change(bank_selection):
    classes = classes_for('Point', 'citizen')
    selection = bank_selection(classes)
    support = selection.for_class(BY_ID[23])[0]
    bank = SupportBank(support.image.parent)
    cv2.imwrite(str(support.image), np.full((20, 30, 3), 201, np.uint8))
    changed = bank.select(classes)
    assert changed.metadata()['manifest_sha256'] == selection.metadata()['manifest_sha256']
    assert changed.metadata()['classes']['citizen'][0]['image_sha256'] != selection.metadata()['classes']['citizen'][0]['image_sha256']


@pytest.mark.parametrize('shots', [0, 2, 6, True])
def test_invalid_shot_count_is_rejected(bank_selection, shots):
    classes = classes_for('Point', 'citizen')
    selection = bank_selection(classes)
    bank = SupportBank(selection.for_class(BY_ID[23])[0].image.parent)
    with pytest.raises(ValueError, match='shots'):
        bank.select(classes, shots)


@pytest.mark.parametrize('mutation', ['too_few', 'too_many', 'repeated_path', 'repeated_id', 'missing_mask',
                                    'empty_mask', 'wrong_size', 'colour_mask', 'duplicate_content'])
def test_invalid_bank_is_rejected_before_model_loading(bank_selection, mutation):
    classes = classes_for('Point', 'citizen')
    selection = bank_selection(classes)
    root = selection.for_class(BY_ID[23])[0].image.parent
    manifest = root / 'bank.json'
    data = json.loads(manifest.read_text())
    entries = data['classes']['citizen']
    if mutation == 'too_few':
        entries.pop()
    elif mutation == 'too_many':
        entries.append(entries[0])
    elif mutation == 'repeated_path':
        entries[1]['image'] = entries[0]['image']
    elif mutation == 'repeated_id':
        entries[1]['id'] = entries[0]['id']
    elif mutation == 'missing_mask':
        (root / entries[0]['mask']).unlink()
    elif mutation == 'empty_mask':
        cv2.imwrite(str(root / entries[0]['mask']), np.zeros((20, 30), np.uint8))
    elif mutation == 'wrong_size':
        cv2.imwrite(str(root / entries[0]['mask']), np.ones((5, 5), np.uint8))
    elif mutation == 'colour_mask':
        cv2.imwrite(str(root / entries[0]['mask']), np.ones((20, 30, 3), np.uint8))
    elif mutation == 'duplicate_content':
        (root / entries[1]['image']).write_bytes((root / entries[0]['image']).read_bytes())
    manifest.write_text(json.dumps(data))
    bank = SupportBank(root)
    assert not bank.validate(classes)['valid']
    with pytest.raises(ValueError):
        bank.select(classes, shots=5)


def test_one_shot_can_run_while_other_slots_are_unfilled(bank_selection):
    classes = classes_for('Point', 'citizen')
    full = bank_selection(classes)
    root = full.for_class(BY_ID[23])[0].image.parent
    for support in full.for_class(BY_ID[23])[1:]:
        support.image.unlink()
        support.mask.unlink()
    bank = SupportBank(root)
    assert len(bank.select(classes, shots=1).for_class(BY_ID[23])) == 1
    assert not bank.validate(classes)['valid']
    with pytest.raises(ValueError):
        bank.select(classes, shots=5)


def test_add_normalizes_indexed_mask_and_never_overwrites(tmp_path):
    root = tmp_path / 'bank'
    initialize_bank(root)
    image, mask = tmp_path / 'raw.png', tmp_path / 'labels.png'
    cv2.imwrite(str(image), np.full((10, 20, 3), 128, np.uint8))
    labels = np.zeros((10, 20), np.uint8)
    labels[2:4, 3:9] = 6
    labels[6:9, 3:9] = 14
    cv2.imwrite(str(mask), labels)
    target = add_example(root, 'citizen', 1, image, mask, foreground_value=6)
    assert sorted(p.name for p in target.iterdir()) == ['image.png', 'mask.png']
    support = SupportBank(root).for_class(BY_ID[23])[0]
    _, binary = support.load()
    assert binary.sum() == 12
    assert set(cv2.imread(str(target / 'mask.png'), 0).ravel()) == {0, 255}
    with pytest.raises(ValueError, match='already exists'):
        add_example(root, 'citizen', 1, image, mask, foreground_value=6)
    with pytest.raises(ValueError, match='duplicate support image'):
        add_example(root, 'citizen', 2, image, mask, foreground_value=6)
    assert not (root / 'citizen/02').exists()
    assert not list(root.rglob('.adding-*'))


def test_selection_only_requires_the_task_class_subset(bank_selection):
    selection = bank_selection(classes_for('Point', 'citizen'))
    bank = SupportBank(selection.for_class(BY_ID[23])[0].image.parent)
    assert bank.validate(classes_for('Point', 'citizen'))['valid']
    assert not bank.validate()['valid']  # Remaining classes not populated.
    with pytest.raises(ValueError, match='absent'):
        selection.for_class(BY_ID[16])
    with pytest.raises(ValueError, match='match'):
        selection.require_classes(classes_for('Polygon', 'road'))


def test_bank_cli_has_no_model_options_and_validate_returns_failure_for_empty_slots(tmp_path, capsys):
    root = tmp_path / 'bank'
    main(['bank', 'init', '--bank', str(root)])
    with pytest.raises(SystemExit) as error:
        main(['bank', 'validate', '--bank', str(root), '--classes', 'citizen'])
    assert error.value.code == 2
    assert '"expected_examples": 5' in capsys.readouterr().out
    args = build_parser().parse_args(['detect', 'flight.mp4', '--output', 'out',
        '--sansa-repo', 'repo', '--checkpoint', 'adapter.pth', '--sam2-checkpoint', 'sam2.pt'])
    assert args.shots == 5 and args.support_bank == 'uav_supports'
    old = build_parser().parse_args(['detect', 'flight.mp4', '--output', 'out',
        '--sansa-repo', 'repo', '--checkpoint', 'adapter.pth', '--sam2-checkpoint', 'sam2.pt',
        '--supports', 'legacy.json', '--shots', '1'])
    assert old.support_bank == 'legacy.json' and old.shots == 1


def test_cli_retrieves_bank_before_constructing_backend(tmp_path, bank_selection, monkeypatch, capsys):
    from triffid_uav_perception import pipeline
    from triffid_uav_perception.models import sansa
    classes = classes_for('Point', 'citizen')
    selection = bank_selection(classes)
    bank_root = selection.for_class(BY_ID[23])[0].image.parent
    constructed = []
    class Stub:
        def __init__(self, *args):
            constructed.append(args)
    def detect(video, srt, output, backend, classes, *, supports, **kwargs):
        assert supports.metadata() == selection.metadata()
        assert len(supports.for_class(BY_ID[23])) == 5
        return 'point_detections.json'
    monkeypatch.setattr(sansa, 'SansaBackend', Stub)
    monkeypatch.setattr(pipeline, 'detect_video', detect)
    argv = ['detect', 'flight.mp4', '--output', 'run', '--classes', 'citizen',
            '--support-bank', str(bank_root), '--sansa-repo', 'repo',
            '--checkpoint', 'adapter_generalist.pth', '--sam2-checkpoint', 'sam2.pt']
    main(argv)
    assert len(constructed) == 1
    assert constructed[0][:3] == ('repo', 'adapter_generalist.pth', 'sam2.pt')
    assert 'point_detections.json' in capsys.readouterr().out
    selection.for_class(BY_ID[23])[0].mask.unlink()
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
    assert len(constructed) == 1  # Invalid bank stopped before another model was loaded.
