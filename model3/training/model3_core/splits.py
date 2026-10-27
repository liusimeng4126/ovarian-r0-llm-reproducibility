"""Reproducible patient-level outer CV and inner validation, using NumPy only."""
import math

import numpy as np


def _check_population(ids, y):
    ids, y = np.asarray(ids), np.asarray(y)
    if ids.ndim != 1 or y.ndim != 1 or len(ids) != len(y):
        raise ValueError("patient IDs and labels must be matching 1-D arrays")
    if not np.issubdtype(ids.dtype, np.integer) or len(set(ids.tolist())) != len(ids):
        raise ValueError("patient IDs must be unique integers")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")
    return ids.astype(np.int64), y.astype(np.int64)


def make_splits(ids, y, seed=42, inner_fraction=0.2):
    ids, y = _check_population(ids, y)
    if not 0.1 <= inner_fraction <= 0.4:
        raise ValueError("inner_fraction must be between 0.1 and 0.4")
    if min(np.bincount(y)) < 5:
        raise ValueError("each class requires at least 5 patients")
    plan = {"seed": int(seed), "n_splits": 5, "inner_fraction": float(inner_fraction), "folds": []}
    outer_folds = _stratified_folds(y, n_splits=5, seed=seed)
    all_indices = np.arange(len(ids), dtype=np.int64)
    for fold, outer_valid in enumerate(outer_folds):
        outer_mask = np.ones(len(ids), dtype=bool)
        outer_mask[outer_valid] = False
        outer_train = all_indices[outer_mask]
        inner_train, inner_valid = _stratified_holdout(
            outer_train, y, fraction=inner_fraction, seed=seed + 1000 + fold,
        )
        plan["folds"].append({
            "fold": fold,
            "outer_train": ids[outer_train].tolist(), "outer_valid": ids[outer_valid].tolist(),
            "inner_train": ids[inner_train].tolist(), "inner_valid": ids[inner_valid].tolist(),
        })
    validate_splits(plan, ids, y)
    return plan


def make_fixed_outer_splits(ids, y, fixed_fold, seed=42, inner_fraction=0.2):
    """Use the supplied 0..4 outer folds and create inner splits from outer-train only."""
    ids, y = _check_population(ids, y)
    fixed_fold = np.asarray(fixed_fold)
    if fixed_fold.shape != ids.shape or not np.issubdtype(fixed_fold.dtype, np.integer):
        raise ValueError("fixed_fold must be one integer per engineering patient")
    if set(np.unique(fixed_fold).tolist()) != set(range(5)):
        raise ValueError("fixed_fold must contain folds 0 through 4")
    if not 0.1 <= inner_fraction <= 0.4:
        raise ValueError("inner_fraction must be between 0.1 and 0.4")
    all_indices = np.arange(len(ids), dtype=np.int64)
    plan = {"seed": int(seed), "n_splits": 5, "inner_fraction": float(inner_fraction),
            "outer_fold_source": "supplied_engineering_fold", "folds": []}
    for fold in range(5):
        outer_valid = np.flatnonzero(fixed_fold == fold).astype(np.int64)
        outer_train = all_indices[fixed_fold != fold]
        inner_train, inner_valid = _stratified_holdout(
            outer_train, y, fraction=inner_fraction, seed=seed + 1000 + fold,
        )
        plan["folds"].append({
            "fold": fold,
            "outer_train": ids[outer_train].tolist(), "outer_valid": ids[outer_valid].tolist(),
            "inner_train": ids[inner_train].tolist(), "inner_valid": ids[inner_valid].tolist(),
        })
    validate_splits(plan, ids, y)
    return plan


def _stratified_folds(y, n_splits, seed):
    """Allocate each class across balanced folds with deterministic shuffling."""
    rng = np.random.RandomState(seed)
    fold_parts = [[] for _ in range(n_splits)]
    labels = sorted(np.unique(y).tolist(), key=lambda value: (-int(np.sum(y == value)), value))
    for label in labels:
        indices = np.flatnonzero(y == label).astype(np.int64)
        rng.shuffle(indices)
        chunks = list(np.array_split(indices, n_splits))
        chunks.sort(key=lambda chunk: -len(chunk))
        destinations = sorted(range(n_splits), key=lambda fold: (sum(map(len, fold_parts[fold])), fold))
        for destination, chunk in zip(destinations, chunks):
            fold_parts[destination].append(chunk)
    folds = []
    for parts in fold_parts:
        combined = np.concatenate(parts).astype(np.int64)
        rng.shuffle(combined)
        folds.append(combined)
    return folds


def _stratified_holdout(parent, y, fraction, seed):
    """Split indices while keeping class counts within one of proportional allocation."""
    parent = np.asarray(parent, dtype=np.int64)
    target = int(math.ceil(len(parent) * fraction))
    labels = np.unique(y[parent])
    counts = np.asarray([np.sum(y[parent] == label) for label in labels], dtype=np.int64)
    if np.any(counts < 2):
        raise ValueError("each class needs at least two patients for an inner holdout")
    ideal = counts.astype(np.float64) * target / len(parent)
    allocation = np.floor(ideal).astype(np.int64)
    allocation = np.maximum(allocation, 1)
    allocation = np.minimum(allocation, counts - 1)
    while int(allocation.sum()) < target:
        available = np.flatnonzero(allocation < counts - 1)
        if not len(available):
            raise ValueError("cannot allocate requested inner validation size")
        choice = available[np.argmax(ideal[available] - allocation[available])]
        allocation[choice] += 1
    while int(allocation.sum()) > target:
        available = np.flatnonzero(allocation > 1)
        if not len(available):
            raise ValueError("cannot allocate requested inner validation size")
        choice = available[np.argmax(allocation[available] - ideal[available])]
        allocation[choice] -= 1

    rng = np.random.RandomState(seed)
    selected = []
    for label, take in zip(labels, allocation):
        indices = parent[y[parent] == label].copy()
        rng.shuffle(indices)
        selected.append(indices[:int(take)])
    valid = np.concatenate(selected).astype(np.int64)
    rng.shuffle(valid)
    valid_set = set(valid.tolist())
    train = np.asarray([index for index in parent.tolist() if index not in valid_set], dtype=np.int64)
    return train, valid


def validate_splits(plan, ids, y):
    ids, y = _check_population(ids, y)
    population = set(ids.tolist())
    labels = dict(zip(ids.tolist(), y.tolist()))
    if plan.get("n_splits") != 5 or len(plan.get("folds", [])) != 5:
        raise ValueError("exactly five outer folds are required")
    fraction = float(plan.get("inner_fraction", -1))
    if not 0.1 <= fraction <= 0.4:
        raise ValueError("invalid inner_fraction")
    fold_numbers, seen = [], []
    for f in plan["folds"]:
        fold_numbers.append(f["fold"])
        groups = {}
        for name in ("outer_train", "outer_valid", "inner_train", "inner_valid"):
            values = f[name]
            if not isinstance(values, list) or any(type(v) is not int for v in values):
                raise ValueError("split patient IDs must be lists of integers")
            groups[name] = set(values)
            if not values or len(groups[name]) != len(values):
                raise ValueError("split contains empty or duplicate patient IDs")
            if not groups[name] <= population:
                raise ValueError("split contains unknown patient IDs")
            if set(labels[v] for v in values) != {0, 1}:
                raise ValueError("every split must contain both classes")
        ot, ov, it, iv = (groups[k] for k in ("outer_train", "outer_valid", "inner_train", "inner_valid"))
        if ot & ov or ot | ov != population:
            raise ValueError("outer train/validation overlap or fail to cover population")
        if it & iv or it | iv != ot:
            raise ValueError("inner train/validation must partition only outer training patients")
        if abs(len(ov) - len(ids) / 5) > 1:
            raise ValueError("outer validation fold size is invalid")
        if abs(len(iv) - len(ot) * fraction) > 1:
            raise ValueError("inner validation size is invalid")
        # Stratified allocations should differ from expected class counts by at most one.
        for child, parent in ((ov, population), (ot, population), (iv, ot), (it, ot)):
            parent_rate = sum(labels[v] for v in parent) / len(parent)
            if abs(sum(labels[v] for v in child) - len(child) * parent_rate) > 1.01:
                raise ValueError("split is not stratified")
        seen.extend(f["outer_valid"])
    if sorted(fold_numbers) != list(range(5)):
        raise ValueError("fold numbers must be 0 through 4 exactly once")
    if len(seen) != len(ids) or set(seen) != population:
        raise ValueError("each patient must appear in outer validation exactly once")


def indices_for_ids(ids, wanted):
    lookup = {int(patient): index for index, patient in enumerate(ids)}
    try:
        return np.asarray([lookup[int(patient)] for patient in wanted], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("split patient ID not present in feature data") from exc
