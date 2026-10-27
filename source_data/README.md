# Aggregate source data

All CSV files are aggregate, non-patient-level sources. Figure 3 files contain performance and five-bin calibration summaries; Figure 4 files contain site coverage, not-mentioned rates, numeric missingness and AUROC change; Figure 5 files contain ablation and 14-site attribution summaries. Patient-level master CSVs and workbooks are excluded.


Precision-recall performance is reported as average precision (AP). Legacy machine-readable field names may use auprc; the reported quantity is average precision computed with sklearn.metrics.average_precision_score.
