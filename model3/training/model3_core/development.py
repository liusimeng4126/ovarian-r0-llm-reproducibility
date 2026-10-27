"""Predeclared development choices, ablations, thresholding and OOF checks."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from .metrics import average_precision_score, roc_auc_score


CANDIDATES: tuple[dict[str, Any], ...] = (
    {"name": "compact_h32", "hidden_dim": 32, "site_embedding_dim": 8,
     "dropout": 0.10, "lr": 1e-3, "weight_decay": 1e-4},
    {"name": "balanced_h48", "hidden_dim": 48, "site_embedding_dim": 12,
     "dropout": 0.20, "lr": 5e-4, "weight_decay": 1e-4},
    {"name": "regularized_h64", "hidden_dim": 64, "site_embedding_dim": 16,
     "dropout": 0.30, "lr": 3e-4, "weight_decay": 1e-3},
)
VARIANTS = ("Full", "noSiteID", "noDelta", "PostOnly", "noAttention")


def candidate_by_name(name: str) -> dict[str, Any]:
    matches = [deepcopy(value) for value in CANDIDATES if value["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"unknown candidate: {name}")
    return matches[0]


def apply_variant(features: Mapping[str, np.ndarray], variant: str) -> dict[str, np.ndarray]:
    """Apply a predeclared input ablation after fold-fitted preprocessing.

    Numeric arrays contain standardized values followed by matching missingness
    masks. Zero standardized value means the training mean; a mask of one marks
    an intentionally unavailable ablated value. Category zero is __MISSING__.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    out = {name: np.asarray(value).copy() for name, value in features.items()}
    if variant in ("Full", "noSiteID", "noAttention"):
        return out
    site = out["site_numeric"]
    glob = out["global_numeric"]
    if site.shape[-1] != 20 or glob.shape[-1] != 24:
        raise ValueError("ablation contract expects 10+10 site and 12+12 global numeric columns")
    if variant == "noDelta":
        site[..., 6:10] = 0.0
        site[..., 16:20] = 1.0
        out["site_categorical"][..., 2] = 0
        # Explicit global deltas and change counts.
        for index in (2, 3, 10, 11):
            glob[..., index] = 0.0
            glob[..., index + 12] = 1.0
        return out
    # PostOnly keeps direct POST measurements/statuses and post site count.
    keep_site = {1, 3, 5}
    for index in range(10):
        if index not in keep_site:
            site[..., index] = 0.0
            site[..., index + 10] = 1.0
    out["site_categorical"][..., 0] = 0
    out["site_categorical"][..., 2] = 0
    keep_global = {1, 5, 7, 9}
    for index in range(12):
        if index not in keep_global:
            glob[..., index] = 0.0
            glob[..., index + 12] = 1.0
    out["global_categorical"][..., 0] = 0
    out["global_categorical"][..., 2] = 0
    return out


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select by mean fold AUROC, then AP, then fewer parameters, then name."""
    if {row.get("candidate") for row in rows} != {c["name"] for c in CANDIDATES}:
        raise ValueError("candidate summary must contain every predeclared candidate exactly")
    if len(rows) != len(CANDIDATES):
        raise ValueError("candidate summary has duplicate rows")
    for row in rows:
        for key in ("mean_fold_auroc", "oof_average_precision", "parameter_count"):
            if key not in row or not np.isfinite(float(row[key])):
                raise ValueError(f"invalid candidate summary field: {key}")
    ranked = sorted(rows, key=lambda r: (-float(r["mean_fold_auroc"]),
                                          -float(r["oof_average_precision"]),
                                          int(r["parameter_count"]), str(r["candidate"])))
    return deepcopy(ranked[0])


def select_youden_threshold(y_true: Any, probability: Any) -> dict[str, Any]:
    """Choose the deterministic OOF threshold maximizing sensitivity+specificity-1.

    Ties choose the threshold closest to 0.5, then the larger threshold. This is
    run once on the selected development OOF predictions, never on locked test.
    """
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("labels and probabilities must be matching non-empty vectors")
    if set(np.unique(y).tolist()) != {0, 1} or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("threshold input must contain two classes and finite probabilities in [0,1]")
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], p)))
    candidates = []
    for threshold in thresholds:
        pred = p >= threshold
        tp = int(np.sum(pred & (y == 1)))
        tn = int(np.sum(~pred & (y == 0)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        candidates.append((sensitivity + specificity - 1.0, -abs(float(threshold) - 0.5),
                           float(threshold), tp, tn, fp, fn, sensitivity, specificity))
    best = max(candidates, key=lambda x: (x[0], x[1], x[2]))
    return {"threshold": best[2], "youden_j": best[0], "sensitivity": best[7],
            "specificity": best[8], "tp": best[3], "tn": best[4],
            "fp": best[5], "fn": best[6], "n": len(y),
            "selection_population": "selected_candidate_292_engineering_oof_only",
            "shadow_used": False, "locked_test_used": False}


def summarize_oof(y_true: Any, probability: Any) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    return {"auroc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "brier": float(np.mean((p - y) ** 2))}


def validate_oof_rows(rows: list[dict[str, Any]], patient_ids: Any, labels: Any) -> None:
    expected = {int(pid): int(y) for pid, y in zip(patient_ids, labels)}
    if len(rows) != len(expected):
        raise ValueError("OOF must contain exactly one row per development patient")
    observed: dict[int, int] = {}
    folds = []
    for row in rows:
        pid, y, fold, probability = int(row["patient_id"]), int(row["y"]), int(row["fold"]), float(row["p_r0"])
        if pid in observed or pid not in expected or y != expected[pid]:
            raise ValueError("OOF patient IDs or labels are duplicated/misaligned")
        if fold not in range(5) or not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("OOF fold/probability is invalid")
        observed[pid] = y
        folds.append(fold)
    if observed != expected or set(folds) != set(range(5)):
        raise ValueError("OOF coverage is incomplete")


def occlude_site(batch: Mapping[str, Any], site_index: int) -> dict[str, Any]:
    """Return a tensor batch with one anatomical site's observed inputs removed."""
    if site_index not in range(14):
        raise ValueError("site_index must be 0..13")
    out = {name: value.clone() for name, value in batch.items()}
    site_dim = out["site_numeric"].shape[-1]
    if site_dim % 2:
        raise ValueError("site numeric values/masks must have equal dimensions")
    half = site_dim // 2
    out["site_numeric"][:, site_index, :half] = 0.0
    out["site_numeric"][:, site_index, half:] = 1.0
    out["site_categorical"][:, site_index, :] = 0
    return out
