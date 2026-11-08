import numpy as np
import pytest

from triffid_uav_perception.evaluation import box_counts, mask_counts, rates


def test_unknown_pixels_are_ignored_without_hiding_annotated_errors():
    gt = np.array([[1, 0], [0, 0]], dtype=bool)
    pred = np.ones((2, 2), bool)
    valid = np.array([[1, 1], [0, 0]], bool)
    assert mask_counts(pred, gt, valid) == dict(tp=1, fp=1, fn=0)
    assert mask_counts(pred, gt) == dict(tp=1, fp=3, fn=0)
    assert rates(mask_counts(pred, gt, valid))['iou'] == .5


def test_empty_unannotated_class_not_scored_as_perfect():
    mask = np.zeros((2, 2), bool)
    assert rates(mask_counts(mask, mask))['iou'] is None
    with pytest.raises(ValueError):
        mask_counts(mask, np.zeros((3, 3), bool))


def test_overlapping_labels_are_evaluated_independently():
    first = np.array([[1, 1], [0, 0]], bool)
    second = np.array([[0, 1], [1, 0]], bool)
    assert rates(mask_counts(first, first))['iou'] == 1
    assert rates(mask_counts(second, second))['iou'] == 1


def test_matching_does_not_count_duplicate_predictions_as_multiple_true_positives():
    box = [0, 0, 10, 10]
    assert box_counts([box, box], [box]) == dict(tp=1, fp=1, fn=0)
    assert box_counts([], [box]) == dict(tp=0, fp=0, fn=1)


def test_box_matching_uses_augmenting_paths_in_ambiguous_case():
    # Broad first prediction can match either GT; second can match only the first.
    assert box_counts([[0, 0, 15, 10], [0, 0, 10, 10]],
                      [[0, 0, 10, 10], [5, 0, 15, 10]]) == dict(tp=2, fp=0, fn=0)
