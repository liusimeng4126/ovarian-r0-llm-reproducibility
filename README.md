# Public reproducibility release (local draft)

Privacy-filtered local release for the ovarian cancer MRI R0 study. It contains reviewed scientific code, reconstructed statistical and attribution wrappers, sanitized Model 3 checkpoints, synthetic input and aggregate CSV sources. It has not been uploaded online.

Figure layout scripts for Figures 3–5 were not found; aggregate sources and final images remain documented in the audit.
\n## Runtime note\nArchived reviewed code requires Python 3.8 or newer; reconstructed wrappers target Python 3.9+.\n

Production structured-extraction model: Qwen3.8-Flash.
The API identifier is `qwen3.8-flash`; pass `--model qwen3.8-flash` with the extractor input/output arguments.


Precision-recall performance is reported as average precision (AP). Legacy machine-readable field names may use auprc; the reported quantity is average precision computed with sklearn.metrics.average_precision_score.
