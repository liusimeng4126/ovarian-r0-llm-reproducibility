"""Binary classification metrics required by Model 3, implemented with NumPy."""
import numpy as np


def _validated(y_true, y_score):
    truth = np.asarray(y_true)
    score = np.asarray(y_score, dtype=np.float64)
    if truth.ndim != 1 or score.ndim != 1 or len(truth) != len(score) or not len(truth):
        raise ValueError("truth and score must be matching non-empty 1-D arrays")
    if not np.isfinite(score).all():
        raise ValueError("scores must be finite")
    if not np.all(np.equal(truth, truth.astype(np.int64))):
        raise ValueError("truth must contain binary integers")
    truth = truth.astype(np.int64)
    if set(np.unique(truth).tolist()) != {0, 1}:
        raise ValueError("both binary classes are required")
    return truth, score


def roc_auc_score(y_true, y_score):
    """Area under the ROC curve with half credit for tied positive/negative scores."""
    truth, score = _validated(y_true, y_score)
    order = np.argsort(score, kind="mergesort")
    truth, score = truth[order], score[order]
    concordant = 0.0
    negatives_below = 0
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and score[end] == score[start]:
            end += 1
        positives = int(truth[start:end].sum())
        negatives = end - start - positives
        concordant += positives * negatives_below + 0.5 * positives * negatives
        negatives_below += negatives
        start = end
    n_positive = int(truth.sum())
    n_negative = len(truth) - n_positive
    return float(concordant / (n_positive * n_negative))


def average_precision_score(y_true, y_score):
    """Non-interpolated average precision, grouping equal scores at one threshold."""
    truth, score = _validated(y_true, y_score)
    order = np.argsort(-score, kind="mergesort")
    truth, score = truth[order], score[order]
    n_positive = int(truth.sum())
    true_positive = 0
    false_positive = 0
    average_precision = 0.0
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and score[end] == score[start]:
            end += 1
        positives = int(truth[start:end].sum())
        negatives = end - start - positives
        true_positive += positives
        false_positive += negatives
        precision = true_positive / (true_positive + false_positive)
        average_precision += precision * (positives / n_positive)
        start = end
    return float(average_precision)
