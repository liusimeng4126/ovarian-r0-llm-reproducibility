# v1.0.1-manuscript

- Corrected the default structured-extraction model identifier to Qwen3.8-Flash to match the actual study configuration.
- Standardized human-readable precision-recall metric terminology to average precision (AP).
- No scientific results, prompts, weights, thresholds, or analysis logic were changed.
- The supplied v1.0.0 currently contained the mixed-case identifier Qwen3.8-Flash; it was normalized to qwen3.8-flash.

# v1.0.0-manuscript

Manuscript reproducibility release. Includes final prompts and structured extraction code, deterministic Model 2 RRS v3.3, Model 3 architecture/training/inference code, three sanitized checkpoints, frozen thresholds, four prespecified ablations, site attribution aggregation, reconstructed statistical reproduction, aggregate figure/table source data, and QC summary.

No patient-level clinical or radiology data are included.

## Model 3 import repair

Restored data.py and model.py as byte-identical archived modules in the complete model3_core package; added local installation configuration and verified imports. No prompts, thresholds, tensors or scientific algorithms changed.

## Final test and portability corrections

Synthetic tests now explicitly disable probability scoring for deterministic RRS and expect AP = 0.5 for the balanced tied-score example. Statistical implementations are unchanged. Local private paths were replaced by required CLI arguments in the checkpoint utility and owner evaluator, and a portable relative path in upload instructions. No scientific results, prompts, weights, thresholds or analysis logic changed.
