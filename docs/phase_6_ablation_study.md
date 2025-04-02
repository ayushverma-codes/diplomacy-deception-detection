# Phase 6 — Verified Power/BERT ablations

> **Final verification (2026-11-08):** The initial lightweight archive did not include Phase 5 predictions or epoch histories. A later, separately supplied `phase6-required-analysis-files.zip` resolved both gaps. In the assembled Phase 1–6 working copy, all 16 Phase 5 prediction CSVs and eight `epochs.jsonl` logs are present. The earlier "missing" statements below are preserved as historical provenance, **not the current artifact status**. See `docs/phase_6_final_verification.md` and `artifacts/phase_6/audit/final_verification.json`. No model retraining, GPU run or fresh checkpoint inference was performed during this final verification.

## Inputs and population consistency

The results below were recomputed from each Phase 5 model's **per-message test CSV**, not copied from prior prose. The eight saved metrics JSON files were independently verified. Across the four architectures, the same task uses precisely the same `game_id`, `message_id`, `absolute_message_index`, and `target` values for every observation. Actual lie: 2,741 test messages, 240 positives; suspected lie: 2,475 annotated messages, 165 positives. Both span test games 4 and 12.

### actual_lie

| architecture      |   samples |   positives | accuracy   | macro_f1   | lie_precision   | lie_recall   | lie_f1   |   predicted_positive |
|:------------------|----------:|------------:|:-----------|:-----------|:----------------|:-------------|:---------|---------------------:|
| Hierarchical LSTM |      2741 |         240 | 89.31%     | 52.63%     | 20.22%          | 7.50%        | 10.94%   |                   89 |
| + Power           |      2741 |         240 | 85.26%     | 57.95%     | 21.92%          | 26.67%       | 24.06%   |                  292 |
| + BERT            |      2741 |         240 | 91.24%     | 47.71%     | 0.00%           | 0.00%        | 0.00%    |                    0 |
| + BERT + Power    |      2741 |         240 | 91.24%     | 47.71%     | 0.00%           | 0.00%        | 0.00%    |                    0 |

### suspected_lie

| architecture      |   samples |   positives | accuracy   | macro_f1   | lie_precision   | lie_recall   | lie_f1   |   predicted_positive |
|:------------------|----------:|------------:|:-----------|:-----------|:----------------|:-------------|:---------|---------------------:|
| Hierarchical LSTM |      2475 |         165 | 90.30%     | 50.93%     | 9.68%           | 5.45%        | 6.98%    |                   93 |
| + Power           |      2475 |         165 | 87.84%     | 53.36%     | 12.64%          | 13.94%       | 13.26%   |                  182 |
| + BERT            |      2475 |         165 | 93.33%     | 48.28%     | 0.00%           | 0.00%        | 0.00%    |                    0 |
| + BERT + Power    |      2475 |         165 | 93.33%     | 48.28%     | 0.00%           | 0.00%        | 0.00%    |                    0 |

## Additive contrasts and interaction (absolute percentage points)

| task          | metric   | power_change   | bert_change   | bert_power_vs_bert   | interaction   |
|:--------------|:---------|:---------------|:--------------|:---------------------|:--------------|
| actual_lie    | macro_f1 | +5.32 pp       | -4.92 pp      | +0.00 pp             | -5.32 pp      |
| actual_lie    | lie_f1   | +13.12 pp      | -10.94 pp     | +0.00 pp             | -13.12 pp     |
| suspected_lie | macro_f1 | +2.43 pp       | -2.65 pp      | +0.00 pp             | -2.43 pp      |
| suspected_lie | lie_f1   | +6.28 pp       | -6.98 pp      | +0.00 pp             | -6.28 pp      |

Power produced a **+5.32 pp** Macro F1 change and **+13.12 pp** positive-class F1 change for actual lies compared with the Phase 5 baseline. For suspected lies it produced **+2.43 pp** Macro F1 and **+6.28 pp** positive-class F1. The Power model detects more lies, but also produces more false positives and lower accuracy. These are observational differences from two held-out games, not a causal proof or statistically reliable population-wide gain.

Both BERT-based variants predicted only class 0 at default threshold and therefore produced exactly the same binary confusion matrices and Macro F1, regardless of Power. Their score distributions differ, meaning equivalent hard-label performance does **not** imply equivalent probability output.

The difference-in-differences interaction `M(BERT+Power) − M(BERT) − M(Power) + M(Baseline)` is descriptive. Its statistical reliability cannot be established from only two independent held-out games.

## Validation-only threshold sensitivity

Validation-derived alternative thresholds and subsequent untouched-test results are saved at `artifacts/phase_6/bert_diagnostics/validation_selected_thresholds.csv`. These are supplementary, separate from original reported metrics; not a new retrained model.
