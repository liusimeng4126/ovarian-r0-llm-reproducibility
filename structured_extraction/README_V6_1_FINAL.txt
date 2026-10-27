V6.1 FINAL package — frozen structured extraction + locked cohort utilities
==========================================================================

A. Locked report subsets
------------------------
1) final_test_100.txt
   - locked final test set, 100 patients / 200 reports
2) development_392.txt
   - all model-development patients, 392 / 784 reports
3) prompt_dev_20.txt
   - extraction-prompt development cases 1-20, 20 / 40 reports
4) model_dev_372_new.txt
   - new model-development cases excluding 1-20, 372 / 744 reports

Report headers are normalized to bare forms such as 148a / 148b. Report body text is preserved.
The original locked split is read from cohort_split_492_locked.xlsx.

B. Extraction V6.1 FINAL
------------------------
File: extract_v6_1_FINAL.py

Key properties:
- The Qwen SYSTEM_PROMPT + PROMPT_TEMPLATE are byte-for-byte logically unchanged from V6.
- Prompt SHA256 remains:
  f364ad0f48e650d9151d46e827b74bc2ce860e6d7e16005f05d40ebb3514139a
- PRE and POST are extracted independently.
- Evidence text is source-backed; Qwen returns clause IDs, Python copies source text.
- Four deterministic V6.1 guards are applied after V6 base validation:
  A distant-metastasis change scope
  B bowel/stomach explicit-object binding
  C mesenteric lymph-node versus mesenteric-peritoneal disease
  D both phases not_mentioned => change not_stated
- No surgical-result label, endpoint classifier, or endpoint scoring logic is present.
- Accepts both 148a and 【148a】 input header styles.

Typical run on all corrected 492 pairs:
  set DASHSCOPE_API_KEY=YOUR_KEY
  python extract_v6_1_FINAL.py --input lca492.txt --outdir v61_492 --overwrite

Or run the development and final-test text files separately if preferred.

Main outputs:
  all_patients_v6_1.jsonl
  review_v6_1.csv
  run_manifest_v6_1.json
  model_trace_v6_1.jsonl
  hotfix_audit_v6_1.csv

C. Independent extraction QC (recommended formal audit)
--------------------------------------------------------
File: qc_v61_independent.py
Locked QC IDs: qc_sample_ids_40_locked.txt

The 40 QC cases were selected with seed 20261012 from model_dev_372_new only.
They exclude prompt-dev 1-20 and final-test 100. Outcome labels are not used.

After the V6.1 batch finishes:
  python qc_v61_independent.py ^
    --jsonl v61_492\all_patients_v6_1.jsonl ^
    --reports model_dev_372_new.txt ^
    --split-xlsx cohort_split_492_locked.xlsx ^
    --ids qc_sample_ids_40_locked.txt ^
    --out qc_v61\qc_review_40.xlsx

A clinician then reviews full PRE/POST reports in the case_reports sheet and fills
manual_* gold-standard columns in site_review.

After saving the filled workbook:
  python qc_v61_independent.py ^
    --score qc_v61\qc_review_40_filled.xlsx ^
    --score-out qc_v61\qc_metrics.json

Important: isolated QC errors are recorded, not used to keep tuning the frozen prompt.
Only a true systematic software/data bug should trigger a pipeline correction and a full rerun.

D. Machine-learning feature tables
----------------------------------
File: build_ml_features_v61.py

Run after V6.1 extraction:
  python build_ml_features_v61.py ^
    --jsonl v61_492\all_patients_v6_1.jsonl ^
    --split-xlsx cohort_split_492_locked.xlsx ^
    --outdir ml_v61

Outputs:
  features_v61_wide.csv
    One row/patient. Recommended Model 2 input.

  site_features_v61_long.csv
    14 rows/patient. Recommended site-wise Model 3 input.

  global_features_v61.csv
    Patient-level global features for Model 3.

  labels_and_split.csv
    Endpoint and locked development/final-test role. Kept separate from feature generation.

  feature_dictionary.csv
  feature_build_manifest.json

Categorical values are intentionally NOT one-hot encoded by this script.
Fit encoders only inside development-set cross-validation folds to avoid leakage.
Model 2 and Model 3 must use the same frozen V6.1 information; only model architecture/data shape differs.

E. Reproducible report splitting
--------------------------------
File: split_lca492_by_locked_cohort.py

  python split_lca492_by_locked_cohort.py ^
    --reports lca492.txt ^
    --split-xlsx cohort_split_492_locked.xlsx ^
    --outdir report_splits

Do not redraw or change the locked final-test set after seeing model performance.


Production structured-extraction model: Qwen3.8-Flash.
The API identifier is `qwen3.8-flash`; pass `--model qwen3.8-flash` with the extractor input/output arguments.
