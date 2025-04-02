# Phase 6 — Research validation and benchmarking

> **Final verification (2026-11-08):** The initial lightweight archive did not include Phase 5 predictions or epoch histories. A later, separately supplied `phase6-required-analysis-files.zip` resolved both gaps. In the assembled Phase 1–6 working copy, all 16 Phase 5 prediction CSVs and eight `epochs.jsonl` logs are present. The earlier "missing" statements below are preserved as historical provenance, **not the current artifact status**. See `docs/phase_6_final_verification.md` and `artifacts/phase_6/audit/final_verification.json`. No model retraining, GPU run or fresh checkpoint inference was performed during this final verification.

## Reproduced from saved prediction artifacts (Phase 2–4 selection)

| task          | model                 |    n | accuracy   | macro_f1   | precision   | recall   | lie_f1   |
|:--------------|:----------------------|-----:|:-----------|:-----------|:------------|:---------|:---------|
| actual_lie    | Phase 4 BiLSTM        | 2741 | 87.12%     | 54.79%     | 19.13%      | 14.58%   | 16.55%   |
| actual_lie    | Phase 2 BoW + Power   | 2741 | 77.13%     | 54.72%     | 16.23%      | 38.75%   | 22.88%   |
| actual_lie    | Phase 4 hierarchical  | 2741 | 88.54%     | 53.32%     | 19.17%      | 9.58%    | 12.78%   |
| actual_lie    | Phase 3 Standard LSTM | 2741 | 85.44%     | 53.12%     | 14.67%      | 13.75%   | 14.19%   |
| actual_lie    | Phase 4 concatenated  | 2741 | 87.41%     | 52.72%     | 15.69%      | 10.00%   | 12.21%   |
| suspected_lie | Phase 4 BiLSTM        | 2475 | 91.03%     | 51.77%     | 12.99%      | 6.06%    | 8.26%    |
| suspected_lie | Phase 3 Standard LSTM | 2475 | 88.08%     | 51.70%     | 9.88%       | 9.70%    | 9.79%    |
| suspected_lie | Phase 4 concatenated  | 2475 | 88.40%     | 51.35%     | 9.33%       | 8.48%    | 8.89%    |
| suspected_lie | Phase 2 BoW + Power   | 2475 | 91.23%     | 51.13%     | 11.76%      | 4.85%    | 6.87%    |
| suspected_lie | Phase 4 hierarchical  | 2475 | 92.36%     | 48.54%     | 3.85%       | 0.61%    | 1.05%    |

## Phase 5 metric-file comparisons

| task          | architecture                 |   n_samples | accuracy   | macro_f1   | precision   | recall   | lie_f1   |
|:--------------|:-----------------------------|------------:|:-----------|:-----------|:------------|:---------|:---------|
| actual_lie    | hierarchical_lstm            |        2741 | 89.31%     | 52.63%     | 20.22%      | 7.50%    | 10.94%   |
| suspected_lie | hierarchical_lstm            |        2475 | 90.30%     | 50.93%     | 9.68%       | 5.45%    | 6.98%    |
| actual_lie    | hierarchical_lstm_power      |        2741 | 85.26%     | 57.95%     | 21.92%      | 26.67%   | 24.06%   |
| suspected_lie | hierarchical_lstm_power      |        2475 | 87.84%     | 53.36%     | 12.64%      | 13.94%   | 13.26%   |
| actual_lie    | hierarchical_lstm_bert       |        2741 | 91.24%     | 47.71%     | 0.00%       | 0.00%    | 0.00%    |
| suspected_lie | hierarchical_lstm_bert       |        2475 | 93.33%     | 48.28%     | 0.00%       | 0.00%    | 0.00%    |
| actual_lie    | hierarchical_lstm_bert_power |        2741 | 91.24%     | 47.71%     | 0.00%       | 0.00%    | 0.00%    |
| suspected_lie | hierarchical_lstm_bert_power |        2475 | 93.33%     | 48.28%     | 0.00%       | 0.00%    | 0.00%    |

The strongest **Phase 5** actual-lie model is Hierarchical LSTM + Power (Macro F1 57.95%, lie F1 24.06%). The same variant leads Phase 5 suspected-lie Macro F1 (53.36%) and lie F1 (13.26%). These Phase 5 metrics agree arithmetically with saved confusion matrices; no per-message recomputation was possible.

## Original ACL 2020 benchmark

Peskov et al. (2020), *It Takes Two to Lie: One to Lie, and One to Listen*, ACL: https://aclanthology.org/2020.acl-main.353/. The paper's figure reports Context LSTM+Power on actual lies at Macro F1 **57.2** and Lie F1 **27.0**, and suspected lies at Macro F1 **53.3**, Lie F1 **13.0**. Our Phase 5 +Power reports actual 57.95/24.06 and suspected 53.36/13.26 (all scores in percent). Proximity of aggregate values is not proof of exact replication: configurations, pretraining strategy, limited context windows, random seeds and metric aggregation may differ.

## Reliability

The dataset contains 12 games total and only **two independent held-out games**. Report game-level variation where per-message data exist, but do not claim reliable statistical significance from 2,741/2,475 correlated message-level examples. Different experiment populations must be matched on message ID before paired comparisons or hypothesis tests.

## Provenance and status

- Phase 2–4 selected results independently recomputed from supplied prediction artifacts.
- Phase 5 reported metrics internally cross-checked against confusion matrices only; per-message scores are missing.
- Original benchmark figures referenced to ACL paper (not regenerated locally).
- No GPU, checkpoint inference, or retraining performed during this Phase 6 analysis.


**Scope and provenance.** Source: user-supplied `phase6-analysis-lightweight.zip` (154 files); derived outputs from `run_phase6_analysis.py`. The original Phase 1–5 project code and model weights were not part of this archive. The eight Phase 5 metric JSON files were checked for internal arithmetic consistency against their confusion matrices, not against absent Phase 5 message-level predictions.


## Follow-up Phase 5 validation

Recomputed all eight Phase 5 test and eight validation confusion matrices and F1 scores from per-message prediction CSVs. All saved JSON metrics matched the recomputation at tight floating-point tolerance. Strict population alignment was verified across the four model architectures for each task/split. Additional epoch histories were available for eight experiments; all BERT best checkpoints were selected at epoch 1, with recorded validation positive-class F1 at zero throughout. Exploratory thresholds use validation data only. These results augment but do not overwrite the earlier research findings.
