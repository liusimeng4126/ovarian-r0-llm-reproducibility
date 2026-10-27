"""Small I/O helpers; no patient data are included in source code manifests."""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def environment():
    versions = {}
    for package in ("numpy", "scipy", "matplotlib", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version, "platform": platform.platform(), "packages": versions,
        "scheduler": "slurm" if os.environ.get("SLURM_JOB_ID") else "not_detected_in_this_process",
    }


def code_manifest():
    root = Path(__file__).resolve().parent.parent
    files = list((root / "model3").glob("*.py"))
    files += list((root / "scripts").glob("*.sh*"))
    files += list((root / "config").glob("*.json"))
    files += list(root.glob("README*.md"))
    files += [p for p in (root / "requirements.txt",) if p.is_file()]
    hashes = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files)}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()
    return {"sha256": digest, "files": hashes}


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
