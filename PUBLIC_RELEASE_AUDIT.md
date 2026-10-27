# Public release audit

Audit scope: recursive review of the local release package, excluding the original project. Raw patient reports, identifiers, patient-level predictions, OOF rows, occlusion rows, traces, credentials and API keys are excluded by construction.

## Packaging checks

- Reviewed final scientific code: FOUND; paths under `structured_extraction/`, `model1_direct_llm/`, `model2_rrs/`, and `model3/`.
- Reconstructed Figure 5 aggregation: FOUND and labeled reconstructed at `ablation_interpretability/aggregate_site_importance.py`; aggregate source CSV is included; historical final wrapper was not found.
- Reconstructed statistical analysis: FOUND and labeled reconstructed at `evaluation/reproduce_final_statistics.py`; it implements AUROC, average precision (AP), Brier, fixed-threshold metrics, 5000 patient-level bootstrap CIs, paired bootstrap Delta AUROC/P values, and five-bin calibration from supplied arrays.
- Model 3 checkpoints: FOUND; three sanitized files for seeds 420042, 420043, 420044. `fit_patient_ids` and private metadata were removed; tensor identity was verified.
- Aggregate sources: FOUND; only non-patient-level Figure 3–5 CSV files are included. Patient-level master CSV and workbook are excluded.
- Figure 3–5 executable layout scripts: NOT FOUND in the source project audit; final images and aggregate sources exist, so exact layout regeneration is not claimed.

## Privacy and credential scan

Recursive filename/content scan found no email addresses, phone numbers, API key assignments, or patient identifier fields in release text/code/data. Checkpoint manifests record only hashes and seeds.

## Scientific limitations

Historical final statistical and Figure 5 wrapper scripts were not recovered; reconstructed code is explicitly labeled and should not be presented as the historical implementation. License, authorship, DOI, ethics and institutional approval remain pending.

## Release decision

PUBLIC RELEASE STATUS: PASS
