# v1.0.1 upload audit

Release: ovarian_r0_llm_public_v1.0.1; version 1.0.1-manuscript.
Source: the relocated ovarian_r0_llm_public_v1.0.0 release. Full source inventory remains byte-identical.

## Authorized corrections

- Production structured-extraction model: Qwen3.8-Flash. DEFAULT_MODEL is qwen3.8-flash. The source used the mixed-case identifier Qwen3.8-Flash.
- Human-readable precision-recall terminology is average precision (AP). Historical machine fields are preserved.
- Restored the complete model3_core Python package using archive-verified byte-identical data.py and model.py and added local installation metadata.
- Fixed two synthetic test mistakes: RRS is scored with probability=False; the tied-score example has AP 0.5.
- Replaced private absolute paths with explicit user-supplied arguments and portable upload instructions.
- No scientific results, prompts, schema, deterministic rules, model tensors, thresholds or statistical algorithms changed.

## Verification

- Python 3.9 syntax: PASS; all release Python files compiled in memory.
- Seven targeted imports including extraction, training and inference: PASS with PyTorch 2.5.1 CPU.
- Five existing synthetic tests: PASS; no clinical study data used.
- Checkpoint identity: PASS for all three seeds; bytes, tensor keys/dtypes/shapes/values match released v1.0.0. Metadata checked for patient IDs and local paths.
- Privacy scan: PASS within scope; no private username/project paths, prohibited patient-level CSV headers, temporary files or nested archives. Generic patient_id/report_text schema names and invented SYNTHETIC_001 are allowed.
- Credential scan: PASS; no detected hard-coded API secrets/bearer tokens; environment-variable reads and placeholder instructions allowed.
- GitHub size: PASS; no file >=100 MB.
- Manifests/filelist/SHA256: rebuilt and verified.
- No external LLM API, model training or upload performed.

This is a targeted packaging validation, not new scientific validation or proof of absence of every possible identifier. Private-data-dependent historical inference is not run end-to-end, and retains original archive input/hash expectations. Author metadata and licensing remain pending before publication; CITATION.cff.template is inactive.

The package contains only released aggregate/synthetic data and sanitized weights. Prior failed import/test checks are resolved. Final file counts and sizes are recorded in provenance/RELEASE_FILELIST.txt.

GITHUB PACKAGE STATUS: PASS
