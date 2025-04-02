# Phase 6 — Initial artifact audit

> **Final verification (2026-11-08):** The initial lightweight archive did not include Phase 5 predictions or epoch histories. A later, separately supplied `phase6-required-analysis-files.zip` resolved both gaps. In the assembled Phase 1–6 working copy, all 16 Phase 5 prediction CSVs and eight `epochs.jsonl` logs are present. The earlier "missing" statements below are preserved as historical provenance, **not the current artifact status**. See `docs/phase_6_final_verification.md` and `artifacts/phase_6/audit/final_verification.json`. No model retraining, GPU run or fresh checkpoint inference was performed during this final verification.

## Inventory and validation

- Archive: 154 files; about 13.93 MB uncompressed.
- Eight Phase 5 `test_metrics.json` and eight validation metric JSON files present; **8/8** test confusion matrices agree with reported accuracy, positive precision/recall/F1, and Macro F1.
- Phase 2 traditional ML and Phase 3–4 test predictions exist and are usable for retrospective analysis. Phase 4 neural training histories, summaries and Colab parity references are present.
- Phase 5 checkpoint **parity-reference JSON files** are present; full `.pt` checkpoints intentionally excluded. Prior local verification of 8/8 parity checks was reported in conversation, but no model execution was performed in this audit.
- **Phase 5 full validation/test prediction CSV files are absent** from the supplied archive.
- **Phase 5 epoch-level history files are absent** from the supplied archive. Best epoch and split-level losses are available in metric JSON files, but these cannot substitute for learning curves.
- Raw `data/` JSONL, all source modules, and complete model weights were omitted by lightweight export design. Source-level and raw-message forensic checks are therefore unavailable here.

## What can be concluded

- Phase 5 held-out test confusion matrices, rates, ablation deltas and all-majority-class BERT *predictions at the 0.5 decision rule* can be checked algebraically.
- Phase 2–4 available prediction CSVs/JSONL permit game-level and disagreement analysis, without original message text.
- Phase 5 per-message probability distributions, per-game outcomes, error examples, probability threshold selection, and training convergence **cannot be independently established** with this export.

## Recommended missing-data transfer

Create a ZIP containing only `artifacts/phase_5/predictions/` and the Phase 5 `history.json` / `training_summary.json` metadata from all eight checkpoint directories. Keep model weights and any original sensitive text out of the ZIP where feasible. Before running threshold tuning, confirm validation and test prediction files contain stable `message_id`, `game_id`, `target`, `prediction`, and `prob_lie` columns. Do not construct invented probability series from aggregate metrics.

## Result consistency

| architecture                 | task          | accuracy   | macro_f1   | lie_f1   |   n_samples |
|:-----------------------------|:--------------|:-----------|:-----------|:---------|------------:|
| hierarchical_lstm            | actual_lie    | 89.31%     | 52.63%     | 10.94%   |        2741 |
| hierarchical_lstm            | suspected_lie | 90.30%     | 50.93%     | 6.98%    |        2475 |
| hierarchical_lstm_power      | actual_lie    | 85.26%     | 57.95%     | 24.06%   |        2741 |
| hierarchical_lstm_power      | suspected_lie | 87.84%     | 53.36%     | 13.26%   |        2475 |
| hierarchical_lstm_bert       | actual_lie    | 91.24%     | 47.71%     | 0.00%    |        2741 |
| hierarchical_lstm_bert       | suspected_lie | 93.33%     | 48.28%     | 0.00%    |        2475 |
| hierarchical_lstm_bert_power | actual_lie    | 91.24%     | 47.71%     | 0.00%    |        2741 |
| hierarchical_lstm_bert_power | suspected_lie | 93.33%     | 48.28%     | 0.00%    |        2475 |


**Scope and provenance.** Source: user-supplied `phase6-analysis-lightweight.zip` (154 files); derived outputs from `run_phase6_analysis.py`. The original Phase 1–5 project code and model weights were not part of this archive. The eight Phase 5 metric JSON files were checked for internal arithmetic consistency against their confusion matrices, not against absent Phase 5 message-level predictions.


## November 2026 — Additional Phase 5 evidence received

The initial audit's missing Phase 5 predictions/training histories have now been supplied in the 56-file `phase6-required-analysis-files.zip`. All 16 prediction CSVs, eight training logs, eight summaries, and 16 metric files are present. All eight test and validation metrics match the supplied per-message predictions and all task/split identifier alignments pass. The initial missing-artifact observation is retained above for provenance; it is **resolved** in this follow-up. Original conversation text, per-message score deltas, and gradient logs were not part of this export.
