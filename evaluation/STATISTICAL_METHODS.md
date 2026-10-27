# Reconstructed statistical methods

This module is a labelled reconstruction for offline public-release analysis. Model 1 uses probability threshold 0.45; Model 2 predicts R0 when RRS is below 2.5 and ranks by negative RRS; Model 3 uses threshold 0.3453282415866852. AUROC, average precision, fixed-threshold sensitivity/specificity/PPV/NPV/F1/accuracy, and Brier score (probability models only) are implemented with SciPy/scikit-learn.

Paired bootstrap comparisons reset `numpy.random.default_rng(20261022)` for each cohort, use 5,000 patient index draws shared by all models, skip one-class draws, report percentile 95% intervals, and calculate two-sided p as `2*min(mean(delta<=0), mean(delta>=0))`, capped at one without a plus-one correction. Calibration sorts stably and splits into five equal-count groups with count, mean probability, event proportion, and extrema.

Site importance computes mean absolute signed delta across the 14 aligned sites before any normalization. No figure layout or patient-level data are included.


Precision-recall performance is reported as average precision (AP). Legacy machine-readable field names may use auprc; the reported quantity is average precision computed with sklearn.metrics.average_precision_score.
