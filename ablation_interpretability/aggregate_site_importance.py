"""Reconstructed aggregate site importance from signed occlusion deltas."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TARGET_RHO = 0.956043956043956
TARGET_P = 9.469782144566512e-08

def aggregate_site_importance(frame: pd.DataFrame, site_columns=None, key="patient_id"):
    if key in frame and frame[key].duplicated().any(): raise ValueError("duplicate alignment keys")
    cols = list(site_columns or [c for c in frame.columns if c.lower().startswith("site")])
    if len(cols) != 14 or not set(cols).issubset(frame.columns): raise ValueError("exactly 14 site columns required")
    x = frame[cols].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(x).all(): raise ValueError("site deltas must be finite")
    mean_abs = np.mean(np.abs(x), axis=0)
    return pd.DataFrame({"site": cols, "mean_abs_delta": mean_abs})

def compare_aggregates(reconstructed: pd.DataFrame, reference: pd.DataFrame):
    a = reconstructed.set_index("site")["mean_abs_delta"]; b = reference.set_index("site")["mean_abs_delta"]
    if set(a.index) != set(b.index): raise ValueError("site columns are not aligned")
    rho, p = spearmanr(a.loc[b.index], b)
    return {"rho": float(rho), "p_value": float(p), "target_rho": TARGET_RHO, "target_p_value": TARGET_P}

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--input", type=Path); ap.add_argument("--aggregate-input", type=Path); ap.add_argument("--outdir", type=Path, required=True)
    args=ap.parse_args(argv)
    if not args.input and not args.aggregate_input: ap.error("provide --input or --aggregate-input")
    df=pd.read_csv(args.input or args.aggregate_input); out=aggregate_site_importance(df) if args.input else df
    args.outdir.mkdir(parents=True, exist_ok=True); out.to_csv(args.outdir/"reconstructed_site_importance.csv", index=False)

if __name__ == "__main__": main()
