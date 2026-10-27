"""Development-package validation and leakage-safe numeric preprocessing.

No supplied Python file is imported or executed. Patient IDs, labels, QC flags
and development subgroup names never enter the feature tensors.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


PACKAGE_FILES = (
    "00_README_FOR_ENGINEER_RRS33.md",
    "01_model3_dev392_structured_clean_NEW_R0.jsonl",
    "02_model3_engineering292_labels_NEW_R0.csv",
    "03_model3_dev392_site_long_NEW_R0.csv",
    "04_model3_shadow100_IDS_NO_LABELS.csv",
    "05_model3_engineering292_structured.jsonl",
    "06_model3_shadow100_structured_NO_LABELS.jsonl",
    "07_model3_dev392_tensor_ready_NEW_R0.npz",
    "08_model3_schema_NEW_R0_RRS33.json",
    "09_MODEL2_RRS33_engineering292_OOF_baseline.csv",
    "10_MODEL2_RRS33_shadow100_scores_NO_LABELS.csv",
    "11_RRS_v3_3_RULES_FROZEN.json",
    "12_MESSAGE_TO_ENGINEER_CN.txt",
    "13_EARLY_STOPPING_QC_CHECKLIST.md",
    "14_validate_package.py",
    "ENVIRONMENT.txt",
    "OWNER_ONLY_RRS33_RESULTS.md",
    "SHA256SUMS.txt",
)
SITE_ORDER = (
    "primary_pelvic_mass", "pelvic_peritoneum", "greater_omentum",
    "lesser_omentum_lesser_sac", "small_bowel_mesenteric", "colonic_mesenteric",
    "paracolic_wall_peritoneum", "diaphragmatic_peritoneum",
    "upper_abdominal_peritoneum", "porta_hepatis", "bowel_stomach_involvement",
    "pelvic_retroperitoneal_nodes", "upper_abdominal_cardiophrenic_nodes",
    "distant_metastasis",
)
STATUS_VOCAB = {
    "__MISSING__": 0, "not_mentioned": 1, "absent": 2,
    "uncertain": 3, "present": 4, "not_evaluable": 5,
}
CHANGE_VOCAB = {
    "__MISSING__": 0, "not_stated": 1, "decreased": 2, "stable": 3,
    "increased": 4, "mixed": 5, "resolved": 6, "new": 7, "not_evaluable": 8,
}
SITE_NUMERIC_NAMES = (
    "burden_pre", "burden_post", "size_pre", "size_post", "relation_pre",
    "relation_post", "burden_delta", "size_delta", "size_rel_delta", "relation_delta",
)
SITE_CATEGORICAL_NAMES = ("pre_status", "post_status", "radiologist_reported_change")
GLOBAL_NUMERIC_NAMES = (
    "peritoneal_size_pre", "peritoneal_size_post", "peritoneal_size_delta",
    "peritoneal_size_rel_delta", "abdo_fluid_grade_pre", "abdo_fluid_grade_post",
    "pleural_grade_pre", "pleural_grade_post", "pre_present_site_count",
    "post_present_site_count", "definite_new_site_count", "definite_resolved_site_count",
)
GLOBAL_CATEGORICAL_NAMES = (
    "abdo_fluid_status_pre", "abdo_fluid_status_post", "pleural_status_pre", "pleural_status_post",
)
GLOBAL_NUMERIC_COLUMNS = GLOBAL_NUMERIC_NAMES
GLOBAL_CATEGORICAL_COLUMNS = GLOBAL_CATEGORICAL_NAMES
_NUMERIC_SOURCES = {
    "site_burden_pre": "pre_burden_grade", "site_burden_post": "post_burden_grade",
    "site_size_pre": "pre_max_size_mm", "site_size_post": "post_max_size_mm",
    "site_relation_pre": "pre_relation_grade", "site_relation_post": "post_relation_grade",
}
_CAT_SOURCES = {
    "site_status_pre": ("pre_status", STATUS_VOCAB),
    "site_status_post": ("post_status", STATUS_VOCAB),
    "site_change": ("radiologist_reported_change", CHANGE_VOCAB),
}
FEATURE_ARRAY_KEYS = {
    "patient_id", *_NUMERIC_SOURCES, *_CAT_SOURCES,
    "global_numeric", "global_categorical",
}
RAW_ARRAY_KEYS = FEATURE_ARRAY_KEYS | {
    "y_engineering_only", "split_code", "engineering_fold",
}
LABEL_ONLY_COLUMNS = {"r0", "outcome_text", "prompt_dev20", "engineering_fold"}
FORBIDDEN_INPUT_TOKENS = {
    "r0", "outcome", "target", "label", "rrs", "pathology", "surgery",
    "evidence", "sentence", "manual_qc", "patient_name", "pinyin",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _numeric_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value)
    _require(arr.shape == shape, f"{name}: expected shape {shape}, got {arr.shape}")
    _require(arr.dtype.kind in "biuf", f"{name}: numeric array required")
    _require(not np.isinf(arr).any(), f"{name}: infinity is invalid; use NaN for missingness")
    with np.errstate(over="ignore", invalid="ignore"):
        result = arr.astype(np.float32)
    _require(not np.isinf(result).any(), f"{name}: values exceed float32 range")
    return result


def _category_array(value: Any, name: str, shape: tuple[int, ...], maxima: Any) -> np.ndarray:
    arr = np.asarray(value)
    _require(arr.shape == shape, f"{name}: expected shape {shape}, got {arr.shape}")
    _require(arr.dtype.kind in "biuf", f"{name}: integer category codes required")
    _require(np.isfinite(arr).all(), f"{name}: categories must be finite")
    _require(np.equal(arr, np.floor(arr)).all(), f"{name}: category codes must be integers")
    _require(((arr >= 0) & (arr <= np.asarray(maxima))).all(), f"{name}: unknown category code")
    return arr.astype(np.int64)


def _patient_ids(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    _require(arr.ndim == 1 and arr.size > 0, "patient_id: nonempty one-dimensional array required")
    _require(arr.dtype.kind in "iu", "patient_id: integer IDs required")
    _require((arr > 0).all(), "patient_id: IDs must be positive")
    _require(np.unique(arr).size == arr.size, "patient_id: duplicate IDs")
    return arr.astype(np.int64)


def _row_indices(indices: Any, n: int, *, unique: bool = False) -> np.ndarray:
    ix = np.asarray(indices)
    _require(ix.ndim == 1, "indices must be one-dimensional")
    if ix.dtype.kind == "b":
        _require(ix.size == n, "boolean indices must match patient count")
        ix = np.flatnonzero(ix)
    _require(ix.dtype.kind in "iu", "indices must be integers or a boolean mask")
    _require(ix.size > 0, "indices must not be empty")
    _require(((ix >= 0) & (ix < n)).all(), "indices are out of bounds")
    _require(not unique or np.unique(ix).size == ix.size, "training indices must be unique")
    return ix.astype(np.int64)


@dataclass
class FeatureBatch:
    patient_id: np.ndarray
    y: np.ndarray | None
    site_numeric: np.ndarray
    site_categorical: np.ndarray
    global_numeric: np.ndarray
    global_categorical: np.ndarray
    site_numeric_names: tuple[str, ...] = SITE_NUMERIC_NAMES
    site_categorical_names: tuple[str, ...] = SITE_CATEGORICAL_NAMES
    global_numeric_names: tuple[str, ...] = GLOBAL_NUMERIC_NAMES
    global_categorical_names: tuple[str, ...] = GLOBAL_CATEGORICAL_NAMES

    def __post_init__(self) -> None:
        self.patient_id = _patient_ids(self.patient_id)
        n = self.patient_id.size
        for attr in ("site_numeric_names", "site_categorical_names", "global_numeric_names", "global_categorical_names"):
            names = tuple(getattr(self, attr))
            _require(len(names) == len(set(names)), f"{attr}: duplicate feature names")
            setattr(self, attr, names)
        _require(self.site_categorical_names == SITE_CATEGORICAL_NAMES, "site categorical channel order differs from contract")
        _require(self.global_categorical_names == GLOBAL_CATEGORICAL_NAMES, "global categorical channel order differs from contract")
        self.site_numeric = _numeric_array(self.site_numeric, "site_numeric", (n, 14, len(self.site_numeric_names)))
        self.site_categorical = _category_array(self.site_categorical, "site_categorical", (n, 14, 3), [5, 5, 8])
        self.global_numeric = _numeric_array(self.global_numeric, "global_numeric", (n, len(self.global_numeric_names)))
        self.global_categorical = _category_array(self.global_categorical, "global_categorical", (n, 4), 5)
        if self.y is not None:
            self.y = _category_array(self.y, "y", (n,), 1).astype(np.int64)

    def subset(self, indices: Any) -> "FeatureBatch":
        ix = _row_indices(indices, self.patient_id.size, unique=True)
        return FeatureBatch(
            self.patient_id[ix], None if self.y is None else self.y[ix],
            self.site_numeric[ix], self.site_categorical[ix],
            self.global_numeric[ix], self.global_categorical[ix],
            self.site_numeric_names, self.site_categorical_names,
            self.global_numeric_names, self.global_categorical_names,
        )

    def metadata(self) -> dict[str, Any]:
        out: dict[str, Any] = {"n_patients": int(self.patient_id.size), "n_sites": 14, "site_order": list(SITE_ORDER)}
        for key in ("site_numeric", "site_categorical", "global_numeric", "global_categorical"):
            out[key + "_names"] = list(getattr(self, key + "_names"))
            out[key + "_shape"] = list(getattr(self, key).shape)
            out[key + "_dim"] = int(getattr(self, key).shape[-1])
        return out


def resolve_package(source: Path, extract_dir: Path) -> Path:
    """Return a package directory or safely extract the exact v2 package.

    The archive may be flat or have one common top-level directory. Unknown
    members, links, duplicate basenames and traversal are rejected. Supplied
    Python is copied as inert source and never imported or executed.
    """
    # Non-strict resolution plus explicit existence checks also works when a
    # Windows sandbox permits file I/O but denies final-path handle queries.
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        return source
    _require(source.is_file() and source.suffix.lower() == ".zip", "source must be a directory or .zip file")
    raw_dest = Path(extract_dir).expanduser()
    _require(not raw_dest.is_symlink(), "extract_dir may not be a symbolic link")
    destination = raw_dest.resolve()
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        files = [entry for entry in members if not entry.is_dir()]
        _require(len(files) == len(PACKAGE_FILES), "ZIP must contain exactly the approved RRS33-v3 package files")
        paths = [PurePosixPath(entry.filename) for entry in files]
        _require(all(not p.is_absolute() and ".." not in p.parts and "\\" not in str(p) for p in paths),
                 "ZIP contains an unsafe path")
        basenames = [p.name for p in paths]
        _require(len(set(basenames)) == len(basenames), "ZIP has duplicate basenames")
        _require(set(basenames) == set(PACKAGE_FILES), "ZIP contains missing or unapproved files")
        parents = {p.parent.as_posix() for p in paths}
        _require(len(parents) == 1 and next(iter(parents)) in (".", "MODEL3_TO_ENGINEER_RRS33_v3"),
                 "ZIP must be flat or use the approved single top-level directory")
        total = 0
        for entry in files:
            path = PurePosixPath(entry.filename)
            mode = entry.external_attr >> 16
            _require(not stat.S_ISLNK(mode), "ZIP symbolic links are forbidden")
            _require(not entry.flag_bits & 1, "encrypted ZIP files are not supported")
            _require(0 <= entry.file_size <= 64 * 1024 * 1024, "ZIP member is unexpectedly large")
            total += entry.file_size
        _require(total <= 128 * 1024 * 1024, "ZIP uncompressed size is unexpectedly large")
        if destination.exists():
            _require(destination.is_dir(), "extract_dir must be a directory")
            for existing in destination.iterdir():
                _require(existing.name in PACKAGE_FILES and existing.is_file() and not existing.is_symlink(),
                         "extract_dir contains unapproved files or links")
        destination.mkdir(parents=True, exist_ok=True)
        for entry in files:
            target = destination / PurePosixPath(entry.filename).name
            _require(target.resolve().parent == destination and not target.is_symlink(), "unsafe extraction target")
            with archive.open(entry, "r") as src, target.open("wb") as dst:
                while block := src.read(1024 * 1024):
                    dst.write(block)
    return destination


def _validate_schema(schema: dict[str, Any]) -> None:
    _require(schema.get("n_development") == 392 and schema.get("n_sites") == 14,
             "schema must describe exactly 392 development patients and 14 sites")
    _require(schema.get("n_engineering") == 292 and schema.get("n_shadow_validation") == 100,
             "schema must describe 292 engineering and 100 shadow patients")
    _require(schema.get("version") in {
        "MODEL3-DATA-CONTRACT-NEW-R0-v2.0",
        "MODEL3-DATA-CONTRACT-NEW-R0-v3.0-RRS33-COMPARATOR",
    },
             "unexpected data contract version")
    _require(schema.get("site_order") == list(SITE_ORDER), "schema site order differs from the fixed 14-site contract")
    _require(schema.get("status_vocab") == STATUS_VOCAB, "schema status vocabulary differs from contract")
    _require(schema.get("change_vocab") == CHANGE_VOCAB, "schema change vocabulary differs from contract")
    _require(schema.get("global_numeric_columns") == list(GLOBAL_NUMERIC_NAMES), "schema global numeric columns differ from whitelist")
    _require(schema.get("global_categorical_columns") == list(GLOBAL_CATEGORICAL_NAMES), "schema global categorical columns differ from whitelist")


def _validate_arrays(arrays: dict[str, np.ndarray]) -> int:
    _require(FEATURE_ARRAY_KEYS <= set(arrays), "NPZ is missing required feature arrays")
    ids = _patient_ids(arrays["patient_id"])
    n = ids.size
    for name in _NUMERIC_SOURCES:
        _numeric_array(arrays[name], name, (n, 14))
        finite = np.asarray(arrays[name])[np.isfinite(arrays[name])]
        if "burden" in name or "relation" in name:
            _require(((finite >= 0) & (finite <= 3) & (finite == np.floor(finite))).all(), f"{name}: grades must be integers 0-3")
        else:
            _require((finite >= 0).all(), f"{name}: sizes cannot be negative")
    for name in _CAT_SOURCES:
        _category_array(arrays[name], name, (n, 14), 8 if name == "site_change" else 5)
    _numeric_array(arrays["global_numeric"], "global_numeric", (n, 12))
    _category_array(arrays["global_categorical"], "global_categorical", (n, 4), 5)
    return n


def _number(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("source contains an invalid numeric field") from exc
    _require(not np.isinf(result), "source numeric field contains infinity")
    return result


def _same_number(a: Any, b: Any) -> bool:
    return bool(np.isclose(_number(a), _number(b), equal_nan=True, atol=1e-5, rtol=1e-5))


def _difference(pre: Any, post: Any, relative: bool = False) -> float:
    pre_n, post_n = _number(pre), _number(post)
    if not np.isfinite(pre_n) or not np.isfinite(post_n) or (relative and pre_n <= 0):
        return float("nan")
    return (post_n - pre_n) / pre_n if relative else post_n - pre_n


def _category(value: Any, vocabulary: dict[str, int]) -> int:
    key = "__MISSING__" if value is None or value == "" else value
    _require(key in vocabulary, "source contains an unknown category")
    return vocabulary[key]


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)), "CSV header is empty or duplicated")
        result = list(reader)
    _require(all(None not in row for row in result), "CSV rows do not match header width")
    return result


def validate_package(package_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Validate hashes, contract, labels and all JSONL/long-table/NPZ values.

    This is strictly a development-data entry point. Test data must use a future
    separate inference importer; it cannot silently enter the 392-case CV pool.
    The report contains aggregate counts only.
    """
    base = Path(package_dir).resolve()
    _require(base.is_dir(), "package_dir must be a directory")
    members = list(base.iterdir())
    _require({p.name for p in members} == set(PACKAGE_FILES), "package directory must contain only the nine development files; test material is forbidden")
    _require(all(p.is_file() and not p.is_symlink() for p in members), "package files must be regular files, not links")
    expected: dict[str, str] = {}
    for line in (base / "SHA256SUMS.txt").read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        _require(len(parts) == 2 and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]) is not None, "invalid SHA256SUMS entry")
        name = parts[1]
        _require(name in PACKAGE_FILES[:-1] and name not in expected, "unapproved or duplicate hash entry")
        expected[name] = parts[0].lower()
    _require(set(expected) == set(PACKAGE_FILES[:-1]), "SHA256SUMS must list all eight payload files")
    for name, wanted in expected.items():
        actual = hashlib.sha256((base / name).read_bytes()).hexdigest()
        _require(actual == wanted, f"SHA256 mismatch: {name}")
    schema = json.loads((base / "05_model3_schema.json").read_text(encoding="utf-8"))
    _validate_schema(schema)
    with np.load(base / "04_model3_dev392_tensor_ready.npz", allow_pickle=False) as archive:
        _require(set(archive.files) == _ARRAY_KEYS, "NPZ array keys differ from the development whitelist")
        arrays = {key: archive[key].copy() for key in archive.files}
    n = _validate_arrays(arrays, require_y=True)
    labels_array = np.asarray(arrays["y"])
    _require(n == 392 and set(np.unique(labels_array).tolist()) == {0, 1},
             "development NPZ must have 392 cases and both binary classes")
    source_rows = [json.loads(line) for line in (base / "01_model3_dev392_structured_clean.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = _csv_rows(base / "02_model3_dev392_labels.csv")
    long_rows = _csv_rows(base / "03_model3_dev392_site_long.csv")
    _require(len(source_rows) == len(labels) == 392 and len(long_rows) == 392 * 14, "source row counts differ from the development contract")
    try:
        source_by_id = {int(r["patient_id"]): r for r in source_rows}
        label_by_id = {int(r["patient_id"]): r for r in labels}
        long_by_pair = {(int(r["patient_id"]), r["site"]): r for r in long_rows}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid source patient_id/site fields") from exc
    ids = set(int(v) for v in arrays["patient_id"])
    _require(len(source_by_id) == len(label_by_id) == 392 and len(long_by_pair) == 5488, "duplicate source patient IDs or site pairs")
    _require(ids == set(source_by_id) == set(label_by_id) == {key[0] for key in long_by_pair}, "patient ID sets disagree across source files")
    _require(set(long_by_pair) == {(pid, site) for pid in ids for site in SITE_ORDER}, "long table must contain all 14 sites exactly once per patient")
    groups = {"prompt_dev20": 0, "model_dev372": 0}
    for label in labels:
        _require(label.get("development_subgroup") in groups, "unknown development group; locked test is forbidden")
        groups[label["development_subgroup"]] += 1
    _require(groups == {"prompt_dev20": 20, "model_dev372": 372}, "development subgroup counts differ from contract")
    for i, pid in enumerate(arrays["patient_id"]):
        row, label = source_by_id[int(pid)], label_by_id[int(pid)]
        _require(label.get("r0") in ("0", "1") and int(label["r0"]) == int(arrays["y"][i]), "labels disagree with NPZ y")
        sites = row.get("sites")
        _require(isinstance(sites, list) and len(sites) == 14 and [s.get("site") for s in sites] == list(SITE_ORDER), "JSONL site count/order differs from contract")
        for j, site in enumerate(sites):
            record = long_by_pair[(int(pid), site["site"])]
            _require(record.get("r0") == label["r0"] and record.get("development_subgroup") == label["development_subgroup"], "long table labels/groups disagree")
            _require(record.get("site_index") == str(j), "long table site_index differs from site order")
            for array_name, field in _NUMERIC_SOURCES.items():
                _require(field in site and _same_number(site[field], arrays[array_name][i, j]) and _same_number(site[field], record.get(field)), f"numeric source correspondence mismatch: {field}")
            for array_name, (field, vocabulary) in _CAT_SOURCES.items():
                _require(field in site and _category(site[field], vocabulary) == arrays[array_name][i, j] and _category(record.get(field), vocabulary) == arrays[array_name][i, j], f"categorical source correspondence mismatch: {field}")
            _require(record.get("status_transition") == f"{site['pre_status']}__TO__{site['post_status']}", "status transition differs from PRE/POST")
            for before, after, field, relative in (
                ("pre_burden_grade", "post_burden_grade", "delta_burden_grade", False),
                ("pre_max_size_mm", "post_max_size_mm", "delta_size_mm", False),
                ("pre_max_size_mm", "post_max_size_mm", "relative_delta_size", True),
                ("pre_relation_grade", "post_relation_grade", "delta_relation_grade", False),
            ):
                _require(_same_number(record.get(field), _difference(site[before], site[after], relative)), f"derived long-table field mismatch: {field}")
        try:
            per, abdo, pleural = row["overall_peritoneal"], row["abdominopelvic_fluid"], row["pleural_effusion"]
            pre_size, post_size = per["pre_group_max_size_mm"], per["post_group_max_size_mm"]
            pre, post = [s["pre_status"] for s in sites], [s["post_status"] for s in sites]
            globals_expected = [pre_size, post_size, _difference(pre_size, post_size), _difference(pre_size, post_size, True), abdo["pre"]["grade"], abdo["post"]["grade"], pleural["pre"]["grade"], pleural["post"]["grade"], sum(v == "present" for v in pre), sum(v == "present" for v in post), sum(a == "absent" and b == "present" for a, b in zip(pre, post)), sum(a == "present" and b == "absent" for a, b in zip(pre, post))]
            cat_expected = [abdo["pre"]["status"], abdo["post"]["status"], pleural["pre"]["status"], pleural["post"]["status"]]
        except (KeyError, TypeError) as exc:
            raise ValueError("JSONL global fields are incomplete") from exc
        for j, value in enumerate(globals_expected):
            _require(_same_number(value, arrays["global_numeric"][i, j]), f"global source correspondence mismatch: {GLOBAL_NUMERIC_NAMES[j]}")
        for j, value in enumerate(cat_expected):
            _require(_category(value, STATUS_VOCAB) == arrays["global_categorical"][i, j], f"global category mismatch: {GLOBAL_CATEGORICAL_NAMES[j]}")
        for record in (long_by_pair[(int(pid), site)] for site in SITE_ORDER):
            numeric_pairs = {
                "overall_peritoneal_pre_group_max_size_mm": pre_size,
                "overall_peritoneal_post_group_max_size_mm": post_size,
                "abdominopelvic_fluid_pre_grade": abdo["pre"]["grade"],
                "abdominopelvic_fluid_post_grade": abdo["post"]["grade"],
                "pleural_effusion_pre_grade": pleural["pre"]["grade"],
                "pleural_effusion_post_grade": pleural["post"]["grade"],
            }
            _require(all(_same_number(record.get(field), value) for field, value in numeric_pairs.items()), "long-table global numeric fields disagree")
            for field, value in zip(("abdominopelvic_fluid_pre_status", "abdominopelvic_fluid_post_status", "pleural_effusion_pre_status", "pleural_effusion_post_status"), cat_expected):
                _require(_category(record.get(field), STATUS_VOCAB) == _category(value, STATUS_VOCAB), "long-table global category fields disagree")
    report = {
        "status": "passed", "sha256_files_verified": 8, "n_development": 392,
        "n_sites": 14, "n_site_rows": 5488, "r0": int(labels_array.sum()),
        "non_r0": int(n - labels_array.sum()),
        "development_groups": groups, "locked_test_present": False,
        "correspondence": "JSONL, labels, long table and NPZ agree by patient/site IDs",
        "numeric_missing_fraction": {key: float(np.isnan(arrays[key]).mean()) for key in (*_NUMERIC_SOURCES, "global_numeric")},
        "category_missing_counts": {key: int((arrays[key] == 0).sum()) for key in (*_CAT_SOURCES, "global_categorical")},
        "provenance_note": "Checksums detect corruption; SHA256SUMS is not an external authenticity signature.",
        "manifest_note": "XLSX manifest and supplied validator are hash checked only; Python source is never executed.",
    }
    return schema, arrays, report


def _recursive_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            current = f"{prefix}.{key}" if prefix else str(key)
            keys.add(current)
            keys.update(_recursive_keys(child, current))
    elif isinstance(value, list):
        for child in value:
            keys.update(_recursive_keys(child, prefix + "[]"))
    return keys


def validate_package_v2(package_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Validate NEW-R0-v2 without executing supplied code or using hidden labels.

    Only NPZ feature arrays and the explicit engineering ``r0`` column can reach
    training. ``outcome_text``, ``rrs_split`` and other label/QC metadata are
    audited but never returned as features.
    """
    base = Path(package_dir).resolve()
    _require(base.is_dir(), "package_dir must be a directory")
    members = list(base.iterdir())
    _require({p.name for p in members} == set(PACKAGE_FILES),
             "package directory differs from the exact NEW-R0 RRS33-v3 whitelist")
    _require(all(p.is_file() and not p.is_symlink() for p in members),
             "package members must be regular files")

    expected: dict[str, str] = {}
    for line in (base / "SHA256SUMS.txt").read_text(encoding="utf-8-sig").splitlines():
        parts = line.split(maxsplit=1)
        _require(len(parts) == 2 and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]) is not None,
                 "invalid SHA256SUMS entry")
        name = parts[1].strip()
        _require(name in PACKAGE_FILES[:-1] and name not in expected,
                 "unapproved or duplicate SHA256SUMS entry")
        expected[name] = parts[0].lower()
    _require(set(expected) == set(PACKAGE_FILES[:-1]),
             f"SHA256SUMS must list all {len(PACKAGE_FILES) - 1} payload files")
    for name, wanted in expected.items():
        actual = hashlib.sha256((base / name).read_bytes()).hexdigest()
        _require(actual == wanted, f"SHA256 mismatch: {name}")

    schema = json.loads((base / "08_model3_schema_NEW_R0_RRS33.json").read_text(encoding="utf-8"))
    _validate_schema(schema)
    with np.load(base / "07_model3_dev392_tensor_ready_NEW_R0.npz", allow_pickle=False) as archive:
        _require(set(archive.files) == RAW_ARRAY_KEYS, "NPZ keys differ from NEW-R0-v2 contract")
        arrays = {key: archive[key].copy() for key in archive.files}
    n = _validate_arrays(arrays)
    _require(n == 392, "NPZ must contain exactly 392 patients")
    split = _category_array(arrays["split_code"], "split_code", (n,), 1)
    y_visible = np.asarray(arrays["y_engineering_only"])
    folds = np.asarray(arrays["engineering_fold"])
    _require(y_visible.shape == folds.shape == (n,), "label/fold arrays have wrong shape")
    _require(np.equal(y_visible, np.floor(y_visible)).all() and np.equal(folds, np.floor(folds)).all(),
             "label/fold arrays must contain integers")
    eng_mask, shadow_mask = split == 0, split == 1
    _require(int(eng_mask.sum()) == 292 and int(shadow_mask.sum()) == 100,
             "split_code must identify 292 engineering and 100 shadow patients")
    _require(set(np.unique(y_visible[eng_mask]).tolist()) == {0, 1}
             and int((y_visible[eng_mask] == 1).sum()) == 122,
             "engineering labels must contain 122 R0 and 170 non-R0")
    _require(np.all(y_visible[shadow_mask] == -1), "shadow labels must remain hidden as -1")
    _require(set(np.unique(folds[eng_mask]).tolist()) == set(range(5))
             and np.all(folds[shadow_mask] == -1),
             "engineering folds must be 0..4 and shadow folds -1")

    labels = _csv_rows(base / "02_model3_engineering292_labels_NEW_R0.csv")
    shadow_ids_rows = _csv_rows(base / "04_model3_shadow100_IDS_NO_LABELS.csv")
    _require(len(labels) == 292 and len(shadow_ids_rows) == 100, "label/ID table row count mismatch")
    _require(set(labels[0]) == {"patient_id", "r0", "engineering_fold", "prompt_dev20", "outcome_text"},
             "engineering label table columns differ from contract")
    _require(set(shadow_ids_rows[0]) == {"patient_id"}, "shadow ID table must contain patient_id only")
    try:
        label_by_id = {int(r["patient_id"]): r for r in labels}
        shadow_ids = {int(r["patient_id"]) for r in shadow_ids_rows}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid patient IDs in labels/shadow tables") from exc
    _require(len(label_by_id) == 292 and len(shadow_ids) == 100 and not (set(label_by_id) & shadow_ids),
             "engineering/shadow IDs are duplicated or overlap")
    ids = arrays["patient_id"].astype(np.int64)
    eng_ids = set(ids[eng_mask].tolist())
    npz_shadow_ids = set(ids[shadow_mask].tolist())
    _require(eng_ids == set(label_by_id) and npz_shadow_ids == shadow_ids,
             "NPZ split IDs disagree with engineering/shadow tables")
    for i, pid in enumerate(ids):
        if eng_mask[i]:
            row = label_by_id[int(pid)]
            _require(row["r0"] in ("0", "1") and int(row["r0"]) == int(y_visible[i]),
                     "engineering CSV and NPZ labels disagree")
            fold_value = float(row["engineering_fold"])
            _require(fold_value.is_integer() and int(fold_value) in range(5)
                     and int(fold_value) == int(folds[i]),
                     "engineering CSV and NPZ folds disagree")

    paths = {
        "all": base / "01_model3_dev392_structured_clean_NEW_R0.jsonl",
        "engineering": base / "05_model3_engineering292_structured.jsonl",
        "shadow": base / "06_model3_shadow100_structured_NO_LABELS.jsonl",
    }
    structured: dict[str, list[dict[str, Any]]] = {}
    for name, path in paths.items():
        structured[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require([len(structured[k]) for k in ("all", "engineering", "shadow")] == [392, 292, 100],
             "structured JSONL row counts differ from contract")
    structured_ids = {name: {int(r["patient_id"]) for r in rows} for name, rows in structured.items()}
    _require(structured_ids["all"] == eng_ids | shadow_ids
             and structured_ids["engineering"] == eng_ids
             and structured_ids["shadow"] == shadow_ids,
             "structured JSONL IDs disagree with the declared partitions")
    for rows in structured.values():
        _require(len({int(r["patient_id"]) for r in rows}) == len(rows), "duplicate JSONL patient ID")
        for row in rows:
            risky = [key for key in _recursive_keys(row)
                     if any(token in key.lower() for token in FORBIDDEN_INPUT_TOKENS)]
            _require(not risky, "structured input contains forbidden outcome/text/QC fields")
            sites = row.get("sites")
            _require(isinstance(sites, list) and [s.get("site") for s in sites] == list(SITE_ORDER),
                     "structured JSONL must contain the fixed 14 sites in order")

    long_rows = _csv_rows(base / "03_model3_dev392_site_long_NEW_R0.csv")
    _require(len(long_rows) == 392 * 14, "long table must contain 392 x 14 rows")
    try:
        long_pairs = {(int(r["patient_id"]), r["site"]) for r in long_rows}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid long-table patient/site fields") from exc
    _require(long_pairs == {(int(pid), site) for pid in ids for site in SITE_ORDER},
             "long table must contain every patient/site pair exactly once")
    for row in long_rows:
        pid = int(row["patient_id"])
        if pid in shadow_ids:
            _require(row.get("r0_engineering_only", "") in ("", "-1"),
                     "shadow label leaked into the long table")

    m2 = _csv_rows(base / "09_MODEL2_RRS33_engineering292_OOF_baseline.csv")
    m2_shadow = _csv_rows(base / "10_MODEL2_RRS33_shadow100_scores_NO_LABELS.csv")
    _require(len(m2) == 292 and len(m2_shadow) == 100, "Model2 comparator row count mismatch")
    _require(set(m2[0]) == {"patient_id", "rrs_score_non_r0", "r0", "engineering_fold",
                            "rrs_rank_score_r0", "pred_r0_at_score_lt_2_5"},
             "Model2 engineering comparator columns differ")
    _require(set(m2_shadow[0]) == {"patient_id", "rrs_score_non_r0", "rrs_rank_score_r0",
                                   "pred_r0_at_score_lt_2_5"},
             "Model2 shadow comparator columns differ or expose labels")
    _require({int(r["patient_id"]) for r in m2} == eng_ids
             and {int(r["patient_id"]) for r in m2_shadow} == shadow_ids,
             "Model2 comparator ID sets disagree")
    for row in m2:
        label = label_by_id[int(row["patient_id"])]
        _require(row["r0"] == label["r0"]
                 and int(float(row["engineering_fold"])) == int(float(label["engineering_fold"])),
                 "Model2 OOF labels/folds disagree with engineering table")
        score = float(row["rrs_score_non_r0"])
        _require(np.isfinite(score) and float(row["rrs_rank_score_r0"]) == -score,
                 "invalid or inconsistent RRS engineering score")
        _require(int(row["pred_r0_at_score_lt_2_5"]) == int(score < 2.5),
                 "RRS engineering frozen-threshold prediction differs")
    for row in m2_shadow:
        score = float(row["rrs_score_non_r0"])
        _require(np.isfinite(score) and float(row["rrs_rank_score_r0"]) == -score,
                 "invalid or inconsistent RRS shadow score")
        _require(int(row["pred_r0_at_score_lt_2_5"]) == int(score < 2.5),
                 "RRS shadow frozen-threshold prediction differs")

    report = {
        "status": "passed", "contract_version": schema["version"],
        "sha256_files_verified": len(PACKAGE_FILES) - 1, "n_total_structured": 392,
        "n_engineering": 292, "n_shadow": 100, "n_locked_internal_test": 0,
        "n_sites": 14, "n_site_rows": 5488, "engineering_r0": 122,
        "engineering_non_r0": 170,
        "fixed_fold_counts": {str(f): int((folds[eng_mask] == f).sum()) for f in range(5)},
        "shadow_labels_visible": False,
        "training_feature_source": "explicit NPZ feature whitelist only",
        "excluded_metadata": sorted(LABEL_ONLY_COLUMNS | {"rrs_split", "r0_engineering_only", "development_subgroup"}),
        "model2_oof_rows_verified": 292, "model2_shadow_rows_verified": 100,
        "locked_test_present": False,
        "supplied_python_executed": False,
        "numeric_missing_fraction": {key: float(np.isnan(arrays[key]).mean())
                                     for key in (*_NUMERIC_SOURCES, "global_numeric")},
        "category_missing_counts": {key: int((arrays[key] == 0).sum())
                                    for key in (*_CAT_SOURCES, "global_categorical")},
    }
    return schema, arrays, report


# Public entry point now refers only to the audited NEW-R0-v2 contract.
validate_package = validate_package_v2


def build_features(arrays: dict[str, np.ndarray], schema: dict[str, Any]) -> FeatureBatch:
    """Use only the explicit numeric/category whitelist; labels remain separate."""
    _validate_schema(schema)
    _validate_arrays(arrays)
    values = [np.asarray(arrays[name], dtype=np.float32) for name in _NUMERIC_SOURCES]
    burden_pre, burden_post, size_pre, size_post, relation_pre, relation_post = values
    def change(pre: np.ndarray, post: np.ndarray, relative: bool = False) -> np.ndarray:
        valid = np.isfinite(pre) & np.isfinite(post)
        if relative:
            valid &= pre > 0
        out = np.full(pre.shape, np.nan, dtype=np.float32)
        np.subtract(post, pre, out=out, where=valid)
        if relative:
            np.divide(out, pre, out=out, where=valid)
        _require(not np.isinf(out).any(), "derived numeric feature overflow")
        return out
    numeric = np.stack(values + [change(burden_pre, burden_post), change(size_pre, size_post), change(size_pre, size_post, True), change(relation_pre, relation_post)], axis=-1)
    categorical = np.stack([arrays[name] for name in _CAT_SOURCES], axis=-1)
    return FeatureBatch(arrays["patient_id"], arrays.get("y"), numeric, categorical,
                        arrays["global_numeric"], arrays["global_categorical"])


class NumericPreprocessor:
    """Fit observed-value statistics exclusively on selected training patients.

    Site statistics pool training patients and their 14 sites, using one shared
    statistic per feature. Values and masks are concatenated on the last axis;
    mask 1 means missing. A column entirely missing in training is held at the
    value 1.0 for all future rows, with the original missingness mask retained.
    """
    def __init__(self) -> None:
        self.params: dict[str, Any] | None = None

    @staticmethod
    def _fit_columns(matrix: np.ndarray, names: tuple[str, ...]) -> dict[str, Any]:
        center, scale, median, observed = [], [], [], []
        for column in matrix.T:
            valid = column[np.isfinite(column)].astype(np.float64)
            observed.append(int(valid.size))
            center.append(float(valid.mean()) if valid.size else 0.0)
            std = float(valid.std(ddof=0)) if valid.size else 1.0
            scale.append(std if std >= 1e-8 else 1.0)
            median.append(float(np.median(valid)) if valid.size else 1.0)
        return {"names": list(names), "mean": center, "std": scale, "median": median,
                "observed_count": observed, "all_missing": [n == 0 for n in observed]}

    def fit(self, batch: FeatureBatch, indices: Any) -> "NumericPreprocessor":
        ix = _row_indices(indices, batch.patient_id.size, unique=True)
        # Validate again in case arrays were mutated after FeatureBatch creation.
        batch.__post_init__()
        self.params = {
            "version": 1, "n_training_patients": int(ix.size), "n_sites": 14,
            "mask_convention": "1=missing, 0=observed; concatenated after numeric values",
            "site_pooling": "training patients x all 14 sites, one statistic per feature",
            "imputation": "training observed median; mean/std use observed training values",
            "all_missing_policy": "constant value 1.0 for every row; preserve original missingness mask; retain feature and site dimensions",
            "site_numeric": self._fit_columns(batch.site_numeric[ix].reshape(-1, batch.site_numeric.shape[-1]), batch.site_numeric_names),
            "global_numeric": self._fit_columns(batch.global_numeric[ix], batch.global_numeric_names),
        }
        return self

    @staticmethod
    def _transform_columns(values: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        missing = np.isnan(values)
        raw = np.where(missing, np.asarray(params["median"], dtype=np.float64), values).astype(np.float64)
        out = (raw - np.asarray(params["mean"])) / np.asarray(params["std"])
        out[..., np.asarray(params["all_missing"], dtype=bool)] = 1.0
        _require(np.isfinite(out).all() and (np.abs(out) <= np.finfo(np.float32).max).all(), "preprocessed features overflow")
        return np.concatenate((out.astype(np.float32), missing.astype(np.float32)), axis=-1)

    def transform(self, batch: FeatureBatch) -> dict[str, np.ndarray]:
        _require(self.params is not None, "NumericPreprocessor must be fitted before transform")
        batch.__post_init__()
        assert self.params is not None
        _require(list(batch.site_numeric_names) == self.params["site_numeric"]["names"] and list(batch.global_numeric_names) == self.params["global_numeric"]["names"], "feature names/order differ from fitted preprocessor")
        return {
            "site_numeric": self._transform_columns(batch.site_numeric, self.params["site_numeric"]),
            "site_categorical": batch.site_categorical.copy(),
            "global_numeric": self._transform_columns(batch.global_numeric, self.params["global_numeric"]),
            "global_categorical": batch.global_categorical.copy(),
        }

    def to_dict(self) -> dict[str, Any]:
        _require(self.params is not None, "NumericPreprocessor has not been fitted")
        return json.loads(json.dumps(self.params, allow_nan=False))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NumericPreprocessor":
        _require(isinstance(payload, dict) and payload.get("version") == 1 and payload.get("n_sites") == 14, "unsupported preprocessor parameters")
        data = json.loads(json.dumps(payload, allow_nan=False))
        for key in ("site_numeric", "global_numeric"):
            section = data.get(key)
            _require(isinstance(section, dict) and isinstance(section.get("names"), list) and len(section["names"]) > 0, "invalid serialized feature names")
            size = len(section["names"])
            _require(len(set(section["names"])) == size, "duplicate serialized feature names")
            for field in ("mean", "std", "median", "observed_count", "all_missing"):
                _require(isinstance(section.get(field), list) and len(section[field]) == size, "invalid serialized parameter dimensions")
            _require(all(type(v) is bool for v in section["all_missing"]), "invalid all_missing flags")
            for field in ("mean", "std", "median", "observed_count"):
                _require(np.isfinite(np.asarray(section[field], dtype=float)).all(), "serialized parameters must be finite")
            _require((np.asarray(section["std"]) > 0).all(), "serialized standard deviations must be positive")
            counts = np.asarray(section["observed_count"])
            _require(((counts >= 0) & (counts == np.floor(counts))).all(), "invalid serialized observed counts")
            _require(np.array_equal(counts == 0, section["all_missing"]), "all_missing flags disagree with observed counts")
        result = cls()
        result.params = data
        return result
