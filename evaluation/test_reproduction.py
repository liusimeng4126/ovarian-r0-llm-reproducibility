import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from reproduce_final_statistics import classification_metrics, calibration_bins, paired_bootstrap
from ablation_interpretability.aggregate_site_importance import aggregate_site_importance

def test_threshold_boundaries():
    y=[0,1]; assert classification_metrics(y,[.45,.9],"M1")["sensitivity"] == 1.0
    result = classification_metrics(y,[2.5,1.0],"M2", probability=False)
    assert result["sensitivity"] == 1.0 and result["specificity"] == 1.0
    assert "brier" not in result

def test_average_precision_ties_and_calibration_stability():
    out=classification_metrics([0,1,1,0],[.2,.2,.8,.8],"M1"); assert out["average_precision"] == .5
    bins=calibration_bins([0,1,0,1],[.5,.5,.5,.5]); assert all(x["count"] == 1 for x in bins)

def test_bootstrap_paired_and_skips():
    out=paired_bootstrap([0,1,0,1],{"a":[.1,.9,.2,.8],"b":[.2,.8,.1,.7]},n_boot=20,pairs=[("a","b")])
    assert out["valid_replicates"] <= 20 and "a-b" in out["pairs"]

def test_signed_cancellation_meanabs_and_rho():
    cols=[f"site{i}" for i in range(14)]; df=pd.DataFrame([[(-1)**i for i in range(14)],[1]*14],columns=cols)
    agg=aggregate_site_importance(df,cols); assert np.allclose(agg.mean_abs_delta,1)
    rho,p=spearmanr(np.arange(14),np.arange(14)); assert rho == 1
