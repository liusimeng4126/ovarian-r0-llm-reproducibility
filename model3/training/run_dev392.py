#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from model3_core.data import build_features, NumericPreprocessor
from model3_core.development import apply_variant, occlude_site, summarize_oof, validate_oof_rows
from model3_core.metrics import roc_auc_score
from model3_core.model import Model3, ModelConfig
from model3_core.splits import make_fixed_outer_splits, indices_for_ids, validate_splits
from model3_core.utils import set_seed

PACKAGE_VERSION = "MODEL3-DEV392-FINAL-2026-10-19-v1"
CANDIDATE = {
    "name": "compact_h32",
    "hidden_dim": 32,
    "site_embedding_dim": 8,
    "dropout": 0.10,
    "lr": 1e-3,
    "weight_decay": 1e-4,
}
VARIANTS = ("Full", "noSiteID", "noDelta", "PostOnly", "noAttention")
EARLY_STOPPING = {
    "monitor": "inner_validation_auroc",
    "mode": "max",
    "patience": 7,
    "max_epochs": 50,
    "min_delta": 1e-12,
    "min_epochs": None,
    "scheduler": None,
}
REQUIRED_BASE_FILES = (
    "07_model3_dev392_tensor_ready_NEW_R0.npz",
    "08_model3_schema_NEW_R0_RRS33.json",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def read_labels_folds(path: Path) -> tuple[dict[int, int], dict[int, int], dict[int, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 392:
        raise ValueError(f"DEVELOPMENT392_LABELS_FOLDS.csv must contain 392 rows; got {len(rows)}")
    labels, folds, sources = {}, {}, {}
    for r in rows:
        pid = int(r["patient_id"])
        y = int(r["r0"])
        fold = int(r["fold"])
        if pid in labels or y not in (0, 1) or fold not in range(5):
            raise ValueError("invalid/duplicate patient_id, r0, or fold in labels file")
        labels[pid], folds[pid], sources[pid] = y, fold, r.get("fold_source", "")
    if sum(labels.values()) != 164:
        raise ValueError("Development392 must contain exactly 164 R0 and 228 non-R0")
    return labels, folds, sources


def _find_member(names: list[str], basename: str) -> str:
    matches = [n for n in names if Path(n).name == basename]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {basename} in base package; found {len(matches)}")
    return matches[0]


def load_base_package(base: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, str]]:
    base = base.expanduser().resolve()
    provenance: dict[str, str] = {"base_package": str(base)}
    if base.is_file() and base.suffix.lower() == ".zip":
        with zipfile.ZipFile(base) as z:
            names = z.namelist()
            npz_member = _find_member(names, REQUIRED_BASE_FILES[0])
            schema_member = _find_member(names, REQUIRED_BASE_FILES[1])
            with np.load(io.BytesIO(z.read(npz_member)), allow_pickle=False) as a:
                arrays = {k: a[k].copy() for k in a.files}
            schema = json.loads(z.read(schema_member).decode("utf-8-sig"))
            provenance.update({
                "base_package_sha256": sha256_file(base),
                "npz_member": npz_member,
                "schema_member": schema_member,
            })
    elif base.is_dir():
        found = {}
        for basename in REQUIRED_BASE_FILES:
            matches = list(base.rglob(basename))
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {basename} under {base}; found {len(matches)}")
            found[basename] = matches[0]
        with np.load(found[REQUIRED_BASE_FILES[0]], allow_pickle=False) as a:
            arrays = {k: a[k].copy() for k in a.files}
        schema = json.loads(found[REQUIRED_BASE_FILES[1]].read_text(encoding="utf-8-sig"))
        provenance.update({
            "npz_path": str(found[REQUIRED_BASE_FILES[0]]),
            "npz_sha256": sha256_file(found[REQUIRED_BASE_FILES[0]]),
            "schema_path": str(found[REQUIRED_BASE_FILES[1]]),
            "schema_sha256": sha256_file(found[REQUIRED_BASE_FILES[1]]),
        })
    else:
        raise FileNotFoundError(f"Base package not found: {base}")
    return arrays, schema, provenance


def prepare(base_package: Path, labels_folds: Path, run_dir: Path, seed: int = 42) -> None:
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"run-dir already exists and is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    arrays, schema, provenance = load_base_package(base_package)
    labels, folds, sources = read_labels_folds(labels_folds)
    pids = np.asarray(arrays["patient_id"], dtype=np.int64)
    if pids.ndim != 1 or len(pids) != 392 or len(set(pids.tolist())) != 392:
        raise ValueError("Base tensor package must contain exactly 392 unique development patient IDs")
    if set(pids.tolist()) != set(labels):
        missing_labels = sorted(set(pids.tolist()) - set(labels))[:20]
        missing_features = sorted(set(labels) - set(pids.tolist()))[:20]
        raise ValueError(f"Patient mismatch. feature-without-label={missing_labels}; label-without-feature={missing_features}")
    y = np.asarray([labels[int(pid)] for pid in pids], dtype=np.int64)
    fixed_fold = np.asarray([folds[int(pid)] for pid in pids], dtype=np.int64)
    feature_arrays = {k: v for k, v in arrays.items() if k in {
        "patient_id", "site_burden_pre", "site_burden_post", "site_size_pre", "site_size_post",
        "site_relation_pre", "site_relation_post", "site_status_pre", "site_status_post", "site_change",
        "global_numeric", "global_categorical"
    }}
    feature_arrays["y"] = y
    batch = build_features(feature_arrays, schema)
    if len(batch.patient_id) != 392 or int(batch.y.sum()) != 164:
        raise ValueError("Prepared feature batch has wrong cohort counts")
    plan = make_fixed_outer_splits(batch.patient_id, batch.y, fixed_fold, seed=seed, inner_fraction=0.2)
    plan["outer_fold_source"] = "extended fixed folds: preserve prior Engineering292 assignments; add former Shadow100 deterministically"
    plan["package_version"] = PACKAGE_VERSION
    plan["candidate_locked"] = CANDIDATE
    plan["variants"] = list(VARIANTS)
    plan["early_stopping"] = EARLY_STOPPING
    validate_splits(plan, batch.patient_id, batch.y)
    np.savez_compressed(
        run_dir / "features392.npz",
        patient_id=batch.patient_id,
        y=batch.y,
        site_numeric=batch.site_numeric,
        site_categorical=batch.site_categorical,
        global_numeric=batch.global_numeric,
        global_categorical=batch.global_categorical,
    )
    write_json(run_dir / "schema.json", schema)
    write_json(run_dir / "splits392.json", plan)
    fold_rows = [{"patient_id": int(pid), "fold": int(folds[int(pid)]), "r0": int(labels[int(pid)]),
                  "fold_source": sources[int(pid)]} for pid in sorted(labels)]
    write_csv(run_dir / "folds392.csv", ["patient_id", "fold", "r0", "fold_source"], fold_rows)
    validation = {
        "status": "passed",
        "package_version": PACKAGE_VERSION,
        "n": 392,
        "r0": int(batch.y.sum()),
        "non_r0": int(len(batch.y) - batch.y.sum()),
        "fold_counts": [
            {"fold": f, "n": int(np.sum(fixed_fold == f)), "r0": int(np.sum(y[fixed_fold == f] == 1)),
             "non_r0": int(np.sum(y[fixed_fold == f] == 0))} for f in range(5)
        ],
        "locked_internal_test_present": False,
        "external_present": False,
        "candidate_reselection_allowed": False,
        "primary_candidate": "compact_h32",
        "provenance": provenance,
    }
    write_json(run_dir / "validation_report.json", validation)
    write_json(run_dir / "prepared_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": PACKAGE_VERSION,
        "features392_sha256": sha256_file(run_dir / "features392.npz"),
        "labels_folds_sha256": sha256_file(labels_folds),
        "folds392_sha256": sha256_file(run_dir / "folds392.csv"),
        "splits392_sha256": sha256_file(run_dir / "splits392.json"),
        "schema_sha256": sha256_file(run_dir / "schema.json"),
        "locked_internal_test_present": False,
    })
    print(json.dumps(validation, ensure_ascii=False, indent=2))


def load_prepared(run_dir: Path):
    manifest = json.loads((run_dir / "prepared_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(run_dir / "features392.npz") != manifest["features392_sha256"]:
        raise ValueError("features392.npz changed after preparation")
    with np.load(run_dir / "features392.npz", allow_pickle=False) as z:
        from model3_core.data import FeatureBatch
        batch = FeatureBatch(*(z[k].copy() for k in (
            "patient_id", "y", "site_numeric", "site_categorical", "global_numeric", "global_categorical")))
    schema = json.loads((run_dir / "schema.json").read_text(encoding="utf-8"))
    plan = json.loads((run_dir / "splits392.json").read_text(encoding="utf-8"))
    validate_splits(plan, batch.patient_id, batch.y)
    if len(batch.patient_id) != 392 or int(batch.y.sum()) != 164:
        raise ValueError("runner requires Development392 with 164 R0 and 228 non-R0")
    return batch, schema, plan


def torch_setup(device_name: str, threads: int):
    if threads < 1:
        raise ValueError("threads must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(threads)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = "cuda" if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available()) else "cpu"
    return torch, torch.device(device)


def tensor_subset(transformed, y, indices, device):
    import torch
    x = {name: torch.as_tensor(value[indices], dtype=torch.long if "categorical" in name else torch.float32,
                               device=device) for name, value in transformed.items()}
    target = torch.as_tensor(y[indices], dtype=torch.float32, device=device)
    return x, target


def train_epoch(model, optimizer, batch, target, batch_size, generator):
    import torch
    model.train()
    order = torch.randperm(len(target), generator=generator)
    total = 0.0
    for start in range(0, len(target), batch_size):
        idx = order[start:start + batch_size].to(target.device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model({k: v[idx] for k, v in batch.items()})
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target[idx])
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.item()) * len(idx)
    avg = total / len(target)
    if not math.isfinite(avg):
        raise ValueError("training failed")
    return avg


def evaluate(model, batch, target):
    import torch
    from model3_core.metrics import average_precision_score, roc_auc_score
    model.eval()
    with torch.inference_mode():
        logits, attention = model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits).cpu().numpy()
        a = attention.cpu().numpy()
    if not np.isfinite(p).all() or not np.isfinite(a).all() or not torch.isfinite(loss):
        raise ValueError("non-finite evaluation output")
    if not np.allclose(a.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("attention weights do not sum to one")
    y = target.cpu().numpy()
    return {"loss": float(loss.item()), "auroc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)), "n": len(y)}, p, a


def plot_history(history, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, prefix in (("Inner training", "train"), ("Inner validation", "valid")):
        axes[0].plot([x["epoch"] for x in history], [x[prefix + "_loss"] for x in history], label=label)
        axes[1].plot([x["epoch"] for x in history], [x[prefix + "_auroc"] for x in history], label=label)
    for ax, title in zip(axes, ("Inner split loss", "Inner split AUROC")):
        ax.set(xlabel="Epoch", title=title); ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def binary_counts(tensor, name):
    values = np.asarray(tensor.detach().cpu().numpy(), dtype=np.int64)
    u, c = np.unique(values, return_counts=True)
    out = {str(int(a)): int(b) for a, b in zip(u, c)}
    if set(out) != {"0", "1"}:
        raise ValueError(f"{name} must contain both classes; got {out}")
    return out


def parameters_changed(model, initial):
    current = dict(model.named_parameters())
    return any(not initial[name].equal(current[name].detach().cpu()) for name in initial)


def code_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted([Path(__file__).resolve(), *sorted((root / "model3_core").glob("*.py"))]):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def train_variant(run_dir: Path, variant: str, device_name: str, epochs: int, patience: int,
                  batch_size: int, threads: int, overwrite: bool = False) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    batch, schema, plan = load_prepared(run_dir)
    torch, device = torch_setup(device_name, threads)
    root = run_dir / "development392_cv" / variant / CANDIDATE["name"]
    if root.exists():
        if not overwrite:
            summary = root / "summary.json"
            status = root / "status.json"
            if summary.is_file() and status.is_file() and json.loads(status.read_text()).get("status") == "complete":
                print(f"SKIP completed: {variant}")
                return json.loads(summary.read_text())
            raise ValueError(f"incomplete output exists: {root}; preserve it or rerun with --overwrite")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    write_json(root / "status.json", {"status": "running", "stage": "three_seed_five_fold_development392_cv"})
    n_seeds = 3
    all_predictions=[]; all_attention=[]; all_occlusion=[]; fold_metrics=[]; seed_oof=[[] for _ in range(n_seeds)]
    all_best_epochs=[]; parameter_count=None
    for fold_spec in plan["folds"]:
        fold=int(fold_spec["fold"]); fold_dir=root/f"fold_{fold}"; fold_dir.mkdir()
        indices={k:indices_for_ids(batch.patient_id,fold_spec[k]) for k in ("inner_train","inner_valid","outer_train","outer_valid")}
        inner_pre=NumericPreprocessor().fit(batch,indices["inner_train"])
        outer_pre=NumericPreprocessor().fit(batch,indices["outer_train"])
        inner_x=apply_variant(inner_pre.transform(batch),variant)
        outer_x=apply_variant(outer_pre.transform(batch),variant)
        config=ModelConfig(site_numeric_dim=inner_x["site_numeric"].shape[-1],global_numeric_dim=inner_x["global_numeric"].shape[-1],
                           hidden_dim=CANDIDATE["hidden_dim"],dropout=CANDIDATE["dropout"],
                           site_embedding_dim=CANDIDATE["site_embedding_dim"],use_attention=variant!="noAttention",use_site_identity=variant!="noSiteID")
        ids=batch.patient_id[indices["outer_valid"]]; labels=batch.y[indices["outer_valid"]]
        per_seed_p=[]; per_seed_a=[]; per_seed_occ=[]; seed_metric_rows=[]
        for seed_index in range(n_seeds):
            seed_dir=fold_dir/f"seed_{seed_index}"; seed_dir.mkdir()
            seed=int(plan["seed"])+fold*10+seed_index
            set_seed(seed)
            model=Model3(config).to(device)
            parameter_count=sum(p.numel() for p in model.parameters())
            if parameter_count>100000: raise ValueError("parameter limit exceeded")
            optimizer=torch.optim.AdamW(model.parameters(),lr=CANDIDATE["lr"],weight_decay=CANDIDATE["weight_decay"])
            generator=torch.Generator(device="cpu").manual_seed(seed)
            train_x,train_y=tensor_subset(inner_x,batch.y,indices["inner_train"],device)
            valid_x,valid_y=tensor_subset(inner_x,batch.y,indices["inner_valid"],device)
            train_class=binary_counts(train_y,"inner_train"); valid_class=binary_counts(valid_y,"inner_valid")
            initial={name:value.detach().cpu().clone() for name,value in model.named_parameters()}
            history=[]; best_auc=-1.0; best_epoch=0; stale=0; best_state=None; stop_reason="max_epochs_reached"; optimizer_updated=False
            for epoch in range(1,epochs+1):
                opt_loss=train_epoch(model,optimizer,train_x,train_y,batch_size,generator)
                if epoch==1:
                    optimizer_updated=parameters_changed(model,initial)
                    if not optimizer_updated: raise ValueError("optimizer did not update parameters at epoch 1")
                tm,_,_=evaluate(model,train_x,train_y); vm,_,_=evaluate(model,valid_x,valid_y)
                if vm["auroc"]>best_auc+1e-12:
                    best_auc=float(vm["auroc"]); best_epoch=epoch; stale=0
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else: stale+=1
                lr=float(optimizer.param_groups[0]["lr"])
                history.append({"fold":fold,"seed_index":seed_index,"seed":seed,"epoch":epoch,"optimization_loss":opt_loss,
                                "train_loss":tm["loss"],"train_auroc":tm["auroc"],"valid_loss":vm["loss"],"valid_auroc":vm["auroc"],
                                "learning_rate":lr,"best_epoch_so_far":best_epoch,"best_score_so_far":best_auc,"patience_counter":stale})
                if stale>=patience:
                    stop_reason="patience_exhausted"; break
            if best_state is None: raise ValueError("early stopping did not select a checkpoint")
            stop_epoch=int(history[-1]["epoch"])
            for row in history: row.update({"selected_best_epoch":best_epoch,"selected_best_score":best_auc,"stop_epoch":stop_epoch,"stop_reason":stop_reason})
            model.load_state_dict(best_state)
            restored,_,_=evaluate(model,valid_x,valid_y)
            if abs(restored["auroc"]-best_auc)>1e-12: raise ValueError("best checkpoint restore mismatch")
            all_best_epochs.append(best_epoch)
            write_csv(seed_dir/"inner_history.csv",list(history[0]),history); plot_history(history,seed_dir/"inner_history.png")
            write_json(seed_dir/"inner_preprocessor.json",inner_pre.to_dict())
            write_json(seed_dir/"early_stopping_qc.json",{**EARLY_STOPPING,"fold":fold,"seed_index":seed_index,"seed":seed,
                       "best_epoch":best_epoch,"best_score":best_auc,"stop_epoch":stop_epoch,"stop_reason":stop_reason,
                       "optimizer_updated_epoch1":optimizer_updated,"train_class_counts":train_class,"valid_class_counts":valid_class,
                       "best_checkpoint_restored_and_verified":True})
            torch.save({"format_version":4,"stage":"inner_development392_only","state_dict":best_state,"model_config":config.to_dict(),
                        "preprocessor":inner_pre.to_dict(),"best_epoch":best_epoch,"best_validation_auroc":best_auc,
                        "fit_patient_ids":fold_spec["inner_train"]},seed_dir/"inner_model.pt")
            # Refit on all outer training patients for selected epoch count.
            refit_seed=seed+100000; set_seed(refit_seed)
            model=Model3(config).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=CANDIDATE["lr"],weight_decay=CANDIDATE["weight_decay"])
            generator=torch.Generator(device="cpu").manual_seed(refit_seed)
            outer_train_x,outer_train_y=tensor_subset(outer_x,batch.y,indices["outer_train"],device)
            refit=[]
            for ep in range(1,best_epoch+1):
                loss=train_epoch(model,optimizer,outer_train_x,outer_train_y,batch_size,generator); refit.append({"epoch":ep,"optimization_loss":loss})
            write_csv(seed_dir/"refit_history.csv",list(refit[0]),refit)
            held_x,held_y=tensor_subset(outer_x,batch.y,indices["outer_valid"],device)
            metric,p,a=evaluate(model,held_x,held_y)
            metric.update({"fold":fold,"seed_index":seed_index,"seed":seed,"best_inner_epoch":best_epoch,"best_inner_auroc":best_auc,
                           "inner_epochs_run":len(history),"stop_epoch":stop_epoch,"stop_reason":stop_reason})
            seed_metric_rows.append(metric); per_seed_p.append(p); per_seed_a.append(a)
            seed_oof[seed_index].extend([{"patient_id":int(pid),"fold":fold,"y":int(y),"p_r0":float(prob)} for pid,y,prob in zip(ids,labels,p)])
            write_csv(seed_dir/"heldout_predictions.csv",["patient_id","fold","seed_index","y","p_r0"],
                      [{"patient_id":int(pid),"fold":fold,"seed_index":seed_index,"y":int(y),"p_r0":float(prob)} for pid,y,prob in zip(ids,labels,p)])
            if variant=="Full":
                occ=[]
                for site_idx in range(14):
                    _,site_p,_=evaluate(model,occlude_site(held_x,site_idx),held_y); occ.append(site_p)
                per_seed_occ.append(np.stack(occ,axis=1))
            checkpoint={"format_version":4,"stage":"development392_outer_fold_seed","state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        "model_config":config.to_dict(),"preprocessor":outer_pre.to_dict(),"candidate":CANDIDATE,"variant":variant,
                        "fold":fold,"seed_index":seed_index,"seed":seed,"refit_seed":refit_seed,"fit_patient_ids":fold_spec["outer_train"],
                        "heldout_patient_ids":fold_spec["outer_valid"],"best_epoch":best_epoch,"prepared_features_sha256":sha256_file(run_dir/"features392.npz")}
            torch.save(checkpoint,seed_dir/"model.pt")
            loaded=torch.load(seed_dir/"model.pt",map_location=device,weights_only=True)
            reload_model=Model3(ModelConfig.from_dict(loaded["model_config"])).to(device); reload_model.load_state_dict(loaded["state_dict"],strict=True)
            _,rp,ra=evaluate(reload_model,held_x,held_y)
            if not np.allclose(rp,p,atol=1e-6,rtol=1e-5) or not np.allclose(ra,a,atol=1e-6,rtol=1e-5):
                raise ValueError("reloaded checkpoint predictions differ")
        ensemble_p=np.mean(np.stack(per_seed_p),axis=0); ensemble_a=np.mean(np.stack(per_seed_a),axis=0)
        fold_stat=summarize_oof(labels,ensemble_p); fold_stat.update({"fold":fold,"candidate":CANDIDATE["name"],"variant":variant,"n":len(labels),"n_seeds":3,"parameter_count":int(parameter_count)})
        fold_metrics.append(fold_stat); write_json(fold_dir/"ensemble_metrics.json",fold_stat); write_csv(fold_dir/"seed_metrics.csv",list(seed_metric_rows[0]),seed_metric_rows)
        fold_pred=[{"patient_id":int(pid),"fold":fold,"y":int(y),"p_r0":float(prob)} for pid,y,prob in zip(ids,labels,ensemble_p)]
        all_predictions.extend(fold_pred); write_csv(fold_dir/"heldout_predictions_ensemble.csv",["patient_id","fold","y","p_r0"],fold_pred)
        att_rows=[{"patient_id":int(pid),"fold":fold,**{name:float(v) for name,v in zip(schema["site_order"],weights)}} for pid,weights in zip(ids,ensemble_a)]
        all_attention.extend(att_rows); write_csv(fold_dir/"heldout_attention_ensemble.csv",["patient_id","fold"]+schema["site_order"],att_rows)
        if variant=="Full":
            ensemble_occ=np.mean(np.stack(per_seed_occ),axis=0)
            occ_rows=[{"patient_id":int(pid),"fold":fold,**{name:float(base_p-site_p) for name,site_p in zip(schema["site_order"],row)}} for pid,base_p,row in zip(ids,ensemble_p,ensemble_occ)]
            all_occlusion.extend(occ_rows); write_csv(fold_dir/"site_occlusion_delta_p_ensemble.csv",["patient_id","fold"]+schema["site_order"],occ_rows)
        write_json(fold_dir/"outer_preprocessor.json",outer_pre.to_dict())
    all_predictions.sort(key=lambda r:int(r["patient_id"])); all_attention.sort(key=lambda r:int(r["patient_id"])); all_occlusion.sort(key=lambda r:int(r["patient_id"]))
    validate_oof_rows(all_predictions,batch.patient_id,batch.y)
    write_csv(root/"oof_predictions.csv",["patient_id","fold","y","p_r0"],all_predictions)
    write_csv(root/"oof_attention.csv",["patient_id","fold"]+schema["site_order"],all_attention)
    if variant=="Full": write_csv(root/"oof_site_occlusion_delta_p.csv",["patient_id","fold"]+schema["site_order"],all_occlusion)
    write_csv(root/"fold_metrics.csv",list(fold_metrics[0]),fold_metrics)
    y=np.asarray([r["y"] for r in all_predictions]); p=np.asarray([r["p_r0"] for r in all_predictions]); oof=summarize_oof(y,p)
    seed_sens=[]
    for idx,rows in enumerate(seed_oof):
        rows.sort(key=lambda r:int(r["patient_id"])); validate_oof_rows(rows,batch.patient_id,batch.y)
        stat=summarize_oof([r["y"] for r in rows],[r["p_r0"] for r in rows]); seed_sens.append({"seed_index":idx,**stat})
    write_csv(root/"single_seed_sensitivity.csv",list(seed_sens[0]),seed_sens)
    summary={"candidate":CANDIDATE["name"],"variant":variant,"n_seeds":3,"mean_fold_auroc":float(np.mean([r["auroc"] for r in fold_metrics])),
             "sd_fold_auroc":float(np.std([r["auroc"] for r in fold_metrics],ddof=1)),"mean_fold_average_precision":float(np.mean([r["average_precision"] for r in fold_metrics])),
             "oof_auroc":oof["auroc"],"oof_average_precision":oof["average_precision"],"oof_brier":oof["brier"],"parameter_count":int(parameter_count),
             "best_epochs":all_best_epochs,"n_oof":len(all_predictions),"early_stopping_protocol":{**EARLY_STOPPING,"patience":patience,"max_epochs":epochs},
             "locked_internal_test_used":False,"external_used":False}
    write_json(root/"summary.json",summary)
    write_json(root/"status.json",{"status":"complete","stage":"three_seed_five_fold_development392_cv","n_oof":392,"all_folds_and_seeds_verified":True})
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def select_threshold(y, p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    candidates=[]
    for t in np.unique(np.concatenate(([0.0,0.5,1.0],p))):
        pred=p>=t; tp=int(np.sum(pred&(y==1))); tn=int(np.sum((~pred)&(y==0))); fp=int(np.sum(pred&(y==0))); fn=int(np.sum((~pred)&(y==1)))
        sens=tp/(tp+fn); spec=tn/(tn+fp); j=sens+spec-1
        candidates.append((j,-abs(float(t)-0.5),float(t),tp,tn,fp,fn,sens,spec))
    b=max(candidates,key=lambda x:(x[0],x[1],x[2]))
    return {"threshold":b[2],"youden_j":b[0],"sensitivity":b[7],"specificity":b[8],"tp":b[3],"tn":b[4],"fp":b[5],"fn":b[6],"n":len(y),"selection_population":"Development392 Full-model OOF only"}


def _fast_auc(y, score):
    y=np.asarray(y,dtype=np.int64); score=np.asarray(score,dtype=np.float64)
    order=np.argsort(score,kind="mergesort"); ys=y[order]; ss=score[order]
    _, first, counts=np.unique(ss,return_index=True,return_counts=True)
    avg_rank=(first+1 + first+counts)/2.0
    ranks_sorted=np.repeat(avg_rank,counts)
    npos=int(ys.sum()); nneg=len(ys)-npos
    if npos==0 or nneg==0: raise ValueError("both classes required")
    return float((np.sum(ranks_sorted[ys==1]) - npos*(npos+1)/2.0)/(npos*nneg))

def bootstrap_delta(y, p_full, p_alt, n_boot=5000, seed=20261019):
    rng=np.random.RandomState(seed); y=np.asarray(y); p_full=np.asarray(p_full); p_alt=np.asarray(p_alt); n=len(y); deltas=[]
    for _ in range(n_boot):
        idx=rng.randint(0,n,n); yb=y[idx]
        if yb.min()==yb.max(): continue
        deltas.append(_fast_auc(yb,p_alt[idx])-_fast_auc(yb,p_full[idx]))
    return {"delta_auroc_alt_minus_full":float(_fast_auc(y,p_alt)-_fast_auc(y,p_full)),
            "ci95_low":float(np.percentile(deltas,2.5)),"ci95_high":float(np.percentile(deltas,97.5)),"bootstrap_valid":len(deltas),"seed":seed}


def aggregate_results(run_dir: Path) -> dict[str, Any]:
    root=run_dir/"development392_cv"; rows=[]; pred_by_variant={}
    for variant in VARIANTS:
        summary=json.loads((root/variant/CANDIDATE["name"]/"summary.json").read_text())
        rows.append({"variant":variant,**summary})
        with (root/variant/CANDIDATE["name"]/"oof_predictions.csv").open(encoding="utf-8-sig",newline="") as f:
            pr=list(csv.DictReader(f))
        pred_by_variant[variant]={int(r["patient_id"]):(int(r["y"]),float(r["p_r0"])) for r in pr}
    full_ids=sorted(pred_by_variant["Full"]); y=np.asarray([pred_by_variant["Full"][i][0] for i in full_ids]); p_full=np.asarray([pred_by_variant["Full"][i][1] for i in full_ids])
    paired=[]
    for variant in VARIANTS[1:]:
        if sorted(pred_by_variant[variant])!=full_ids: raise ValueError("ablation patient IDs differ")
        p_alt=np.asarray([pred_by_variant[variant][i][1] for i in full_ids]); paired.append({"variant":variant,**bootstrap_delta(y,p_full,p_alt)})
    threshold=select_threshold(y,p_full)
    write_csv(root/"ablation_comparison392.csv",["variant","candidate","n_seeds","mean_fold_auroc","sd_fold_auroc","mean_fold_average_precision","oof_auroc","oof_average_precision","oof_brier","parameter_count","n_oof"],
              [{k:r.get(k) for k in ["variant","candidate","n_seeds","mean_fold_auroc","sd_fold_auroc","mean_fold_average_precision","oof_auroc","oof_average_precision","oof_brier","parameter_count","n_oof"]} for r in rows])
    write_csv(root/"paired_bootstrap_vs_full392.csv",["variant","delta_auroc_alt_minus_full","ci95_low","ci95_high","bootstrap_valid","seed"],paired)
    write_json(root/"FULL_THRESHOLD_FROZEN_FROM_DEV392_OOF.json",threshold)
    result={"package_version":PACKAGE_VERSION,"primary_model":"Full compact_h32 remains primary; ablations are mechanistic and are NOT eligible to replace it after seeing these results.",
            "rows":rows,"paired_bootstrap_vs_full":paired,"full_threshold":threshold,"locked_internal_test_used":False,"external_used":False}
    write_json(root/"MODEL3_DEVELOPMENT392_RESULTS.json",result)
    return result


def fit_final(run_dir: Path, device_name: str, batch_size: int, threads: int) -> dict[str, Any]:
    import torch
    batch,schema,_=load_prepared(run_dir); device_torch,device=torch_setup(device_name,threads)
    result_path=run_dir/"development392_cv"/"MODEL3_DEVELOPMENT392_RESULTS.json"
    if not result_path.is_file(): raise ValueError("run all five variants and aggregate before final fit")
    result=json.loads(result_path.read_text()); full_summary=next(r for r in result["rows"] if r["variant"]=="Full")
    epochs=[max(1,int(round(median(full_summary["best_epochs"][i::3])))) for i in range(3)]
    out=run_dir/"FROZEN_MODEL3_AFTER_DEV392"
    if out.exists(): raise ValueError("freeze directory already exists; do not overwrite a frozen model")
    out.mkdir()
    pre=NumericPreprocessor().fit(batch,np.arange(len(batch.patient_id))); transformed=apply_variant(pre.transform(batch),"Full")
    config=ModelConfig(site_numeric_dim=transformed["site_numeric"].shape[-1],global_numeric_dim=transformed["global_numeric"].shape[-1],
                       hidden_dim=CANDIDATE["hidden_dim"],dropout=CANDIDATE["dropout"],site_embedding_dim=CANDIDATE["site_embedding_dim"],use_attention=True,use_site_identity=True)
    x,y=tensor_subset(transformed,batch.y,np.arange(len(batch.patient_id)),device); members=[]
    for seed_index,epoch_count in enumerate(epochs):
        seed=420042+seed_index; set_seed(seed); model=Model3(config).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=CANDIDATE["lr"],weight_decay=CANDIDATE["weight_decay"]); generator=torch.Generator(device="cpu").manual_seed(seed)
        history=[]
        for ep in range(1,epoch_count+1): history.append({"epoch":ep,"optimization_loss":train_epoch(model,optimizer,x,y,batch_size,generator)})
        sd=out/f"seed_{seed_index}"; sd.mkdir(); write_csv(sd/"training_history.csv",list(history[0]),history)
        checkpoint={"format_version":5,"stage":"frozen_after_development392_before_internal_test","state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    "model_config":config.to_dict(),"preprocessor":pre.to_dict(),"candidate":CANDIDATE,"variant":"Full","threshold":result["full_threshold"],
                    "epochs":epoch_count,"seed_index":seed_index,"seed":seed,"site_order":schema["site_order"],"fit_patient_ids":batch.patient_id.tolist(),
                    "development392_features_sha256":sha256_file(run_dir/"features392.npz"),"locked_internal_test_used":False,"external_used":False}
        torch.save(checkpoint,sd/"model.pt"); members.append({"seed_index":seed_index,"seed":seed,"epochs":epoch_count,"checkpoint_sha256":sha256_file(sd/"model.pt")})
    write_json(out/"preprocessor.json",pre.to_dict()); write_csv(out/"ensemble_members.csv",list(members[0]),members)
    freeze={"status":"FROZEN_BEFORE_INTERNAL_TEST","created_utc":datetime.now(timezone.utc).isoformat(),"package_version":PACKAGE_VERSION,
            "candidate":CANDIDATE,"variant":"Full","n_models":3,"members":members,"threshold":result["full_threshold"],"development392_results_sha256":sha256_file(result_path),
            "features392_sha256":sha256_file(run_dir/"features392.npz"),"folds392_sha256":sha256_file(run_dir/"folds392.csv"),"code_sha256":code_hash(Path(__file__).resolve().parent),
            "locked_internal_test_used":False,"external_used":False,"next_step":"Generate probabilities only on the untouched internal-test100 with labels withheld; evaluate owner-side after prediction hash is frozen."}
    write_json(out/"FREEZE_MANIFEST.json",freeze)
    print(json.dumps(freeze,ensure_ascii=False,indent=2)); return freeze


def main():
    ap=argparse.ArgumentParser(description="Final Development392 rerun for Model3 after Engineering292 selection/QC.")
    ap.add_argument("--base-package", help="Original MODEL3_TO_ENGINEER_RRS33_v3_PACKAGE.zip or extracted directory")
    ap.add_argument("--labels-folds", default=str(Path(__file__).resolve().parent/"DEVELOPMENT392_LABELS_FOLDS.csv"))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--patience", type=int, default=7); ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--device", choices=("auto","cpu","cuda"), default="auto")
    ap.add_argument("--overwrite-variant", action="store_true"); ap.add_argument("--no-final-fit", action="store_true")
    args=ap.parse_args(); run_dir=Path(args.run_dir).expanduser().resolve(); labels=Path(args.labels_folds).expanduser().resolve()
    if not args.train_only:
        if not args.base_package: raise ValueError("--base-package is required unless --train-only is used")
        prepare(Path(args.base_package),labels,run_dir)
        if args.prepare_only: return
    for variant in args.variants: train_variant(run_dir,variant,args.device,args.epochs,args.patience,args.batch_size,args.threads,args.overwrite_variant)
    if set(args.variants)==set(VARIANTS):
        result=aggregate_results(run_dir); print(json.dumps(result,ensure_ascii=False,indent=2))
        if not args.no_final_fit: fit_final(run_dir,args.device,args.batch_size,args.threads)
    else:
        print("Partial variant run finished; aggregate/final freeze skipped because not all five prespecified variants were requested.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("INTERRUPTED. Preserve the run directory; completed variants may be resumed.", file=sys.stderr); raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise
