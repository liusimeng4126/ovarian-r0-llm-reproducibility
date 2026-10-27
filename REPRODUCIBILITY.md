# Reproducibility

Use Python 3.9 or newer and install `requirements.txt`. Reviewed archived scientific code is marked original in the provenance manifest. `evaluation/reproduce_final_statistics.py` and `ablation_interpretability/aggregate_site_importance.py` are reconstructed reproduction scripts and are explicitly not claimed to be historical wrappers. Tests use synthetic or aggregate data only; no external LLM calls or retraining are required.

Model 1 uses P(R0) >= 0.45 selected by maximum Youden J on Engineering292 (n=292). Model 2 uses RRS < 2.5 for R0. Model 3 uses seeds 420042, 420043, 420044 and threshold 0.3453282415866852 from Development392 Full five-fold OOF probabilities.
