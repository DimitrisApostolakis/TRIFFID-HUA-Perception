"""Small held-out mask/box metrics; no model imports or threshold tuning."""
import numpy as np


def mask_counts(prediction, truth, valid=None):
    prediction, truth = np.asarray(prediction, dtype=bool), np.asarray(truth, dtype=bool)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError('Predicted and annotated masks must share original image dimensions')
    if valid is None:
        valid = np.ones_like(truth)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != truth.shape:
        raise ValueError('Evaluation region dimensions differ')
    return dict(tp=int((prediction & truth & valid).sum()),
                fp=int((prediction & ~truth & valid).sum()),
                fn=int((~prediction & truth & valid).sum()))


def rates(counts):
    tp, fp, fn = (counts[key] for key in ('tp', 'fp', 'fn'))
    return dict(iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None)


def box_counts(predictions, truth, threshold=.5):
    """Maximum-cardinality one-to-one matching, independent of null/tied scores."""
    if not 0 < threshold <= 1:
        raise ValueError('IoU threshold must lie in (0,1]')
    def overlap(a, b):
        intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
        union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
        return intersection/union if union > 0 else 0
    edges = [sorted([i for i,b in enumerate(truth) if overlap(a,b) >= threshold],
                    key=lambda i: -overlap(a,truth[i])) for a in predictions]
    owners = {}
    def assign(index, seen):
        for target in edges[index]:
            if target in seen:
                continue
            seen.add(target)
            if target not in owners or assign(owners[target], seen):
                owners[target] = index
                return True
        return False
    for index in range(len(predictions)):
        assign(index, set())
    return dict(tp=len(owners), fp=len(predictions)-len(owners), fn=len(truth)-len(owners))
