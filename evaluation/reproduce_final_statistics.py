"""Reconstructed, offline final evaluation statistics for the public release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

THRESHOLDS = {"M1": 0.45, "M2": 2.5, "M3": 0.3453282415866852}


def _arrays(y_true, score):
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or len(y) == 0:
        raise ValueError("labels and scores must be non-empty one-dimensional arrays of equal length")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    vals = set(np.unique(y).tolist())
    if not vals.issubset({0, 1}) or len(vals) < 2:
        raise ValueError("labels must contain both binary classes 0 and 1")
    return y.astype(int), s


def classification_metrics(y_true, score, model: str, *, probability: bool = True) -> Dict[str, float]:
    """Return discrimination, calibration (probability models), and frozen-threshold metrics."""
    y, s = _arrays(y_true, score)
    if model == "M2":
        pred = s < THRESHOLDS[model]
        ranking = -s
    else:
        pred = s >= THRESHOLDS[model]
        ranking = s
    tp = int(np.sum((pred == 1) & (y == 1))); tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0))); fn = int(np.sum((pred == 0) & (y == 1)))
    div = lambda a, b: float(a / b) if b else float("nan")
    out = {"auroc": float(roc_auc_score(y, ranking)), "average_precision": float(average_precision_score(y, ranking))}
    out.update({"sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp), "ppv": div(tp, tp + fp),
                "npv": div(tn, tn + fn), "f1": div(2 * tp, 2 * tp + fp + fn), "accuracy": float(np.mean(pred == y))})
    if probability:
        if np.any((s < 0) | (s > 1)):
            raise ValueError("probabilities must lie in [0, 1]")
        out["brier"] = float(brier_score_loss(y, s))
    return out


def calibration_bins(y_true, probability, groups: int = 5):
    y, p = _arrays(y_true, probability)
    order = np.argsort(p, kind="mergesort")
    result = []
    for idx in np.array_split(order, groups):
        if len(idx) == 0: continue
        result.append({"count": int(len(idx)), "mean_probability": float(np.mean(p[idx])),
                       "event_proportion": float(np.mean(y[idx])), "min_probability": float(np.min(p[idx])),
                       "max_probability": float(np.max(p[idx]))})
    return result


def paired_bootstrap(y_true, scores: Mapping[str, Sequence[float]], metric: Callable = roc_auc_score,
                     *, seed: int = 20261022, n_boot: int = 5000, pairs: Iterable[Tuple[str, str]] = ()):
    y = np.asarray(y_true); n = len(y)
    if n == 0 or not set(np.unique(y).tolist()).issubset({0, 1}): raise ValueError("invalid binary labels")
    arrays = {k: np.asarray(v, dtype=float) for k, v in scores.items()}
    if any(len(v) != n or not np.all(np.isfinite(v)) for v in arrays.values()): raise ValueError("invalid scores")
    rng = np.random.default_rng(seed); estimates = {k: float(metric(y, v)) for k, v in arrays.items()}
    draws = {k: [] for k in arrays}; deltas = {pair: [] for pair in pairs}; skipped = 0
    for _ in range(n_boot):
        ix = rng.integers(0, n, n); yy = y[ix]
        if len(np.unique(yy)) < 2: skipped += 1; continue
        vals = {k: float(metric(yy, v[ix])) for k, v in arrays.items()}
        for k, val in vals.items(): draws[k].append(val)
        for a, b in deltas: deltas[(a, b)].append(vals[a] - vals[b])
    result = {"valid_replicates": len(next(iter(draws.values()))), "skipped_replicates": skipped,
              "estimates": estimates, "ci": {k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in draws.items()}, "pairs": {}}
    for pair, vals in deltas.items():
        d = np.asarray(vals); result["pairs"][f"{pair[0]}-{pair[1]}"] = {"delta": estimates[pair[0]] - estimates[pair[1]],
            "ci": tuple(np.percentile(d, [2.5, 97.5])), "p": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))}
    return result


def evaluate_csv(path: Path, outdir: Path, label_col="label", id_col="patient_id"):
    df = pd.read_csv(path)
    if label_col not in df: raise ValueError("missing label column")
    if id_col in df and df[id_col].duplicated().any(): raise ValueError("duplicate record keys")
    models = {m: next((c for c in df.columns if c.lower() in {m.lower(), f"{m.lower()}_score", f"{m.lower()}_probability"}), None) for m in THRESHOLDS}
    if any(v is None for v in models.values()): raise ValueError("missing model score column")
    out = {m: classification_metrics(df[label_col], df[c], m, probability=(m != "M2")) for m, c in models.items()}
    outdir.mkdir(parents=True, exist_ok=True); (outdir / "reconstructed_statistics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--input", type=Path, required=True); ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args(argv); evaluate_csv(args.input, args.outdir)


if __name__ == "__main__": main()
