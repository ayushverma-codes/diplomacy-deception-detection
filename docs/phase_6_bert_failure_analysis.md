# Phase 6 — BERT failure analysis (updated with per-message evidence)

> **Final verification (2026-11-08):** The initial lightweight archive did not include Phase 5 predictions or epoch histories. A later, separately supplied `phase6-required-analysis-files.zip` resolved both gaps. In the assembled Phase 1–6 working copy, all 16 Phase 5 prediction CSVs and eight `epochs.jsonl` logs are present. The earlier "missing" statements below are preserved as historical provenance, **not the current artifact status**. See `docs/phase_6_final_verification.md` and `artifacts/phase_6/audit/final_verification.json`. No model retraining, GPU run or fresh checkpoint inference was performed during this final verification.

## Evidence and validation

This revision uses **all 16 Phase 5 prediction CSVs** (eight model/task pairs × validation/test), eight test and eight validation metric JSON files, eight epoch logs, and the training summaries from the uploaded Drive export. Recomputed accuracy, Macro F1, positive-class F1 and confusion matrices **match every saved metrics JSON**. All architectures have identical message identifiers and ground-truth labels within each task and split. Original reference threshold = model's saved predictions (approximately 0.5). No models were retrained or evaluated using a test-selected threshold.

## Verified failure mode

All four BERT experiments predicted **zero positive labels** on both validation and test at the original threshold: actual lie (240/2,741 positives in test) and suspected lie (165/2,475). The BERT models' test probability scores vary, but occupy relatively narrow ranges, entirely below 0.5. This supports **decision-level negative-class collapse**, not the stronger claim that all internal BERT representations are numerically identical.

### Probability/ranking diagnostics — original test predictions

| architecture   | task          |    min |    max |   mean |    std |   mean_positive |   mean_negative |   average_precision |   roc_auc |
|:---------------|:--------------|-------:|-------:|-------:|-------:|----------------:|----------------:|--------------------:|----------:|
| + BERT         | actual_lie    | 0.2176 | 0.2609 | 0.224  | 0.0142 |          0.2241 |          0.224  |              0.09   |    0.51   |
| + BERT + Power | actual_lie    | 0.1079 | 0.1901 | 0.1276 | 0.0173 |          0.1287 |          0.1275 |              0.1064 |    0.5599 |
| + BERT         | suspected_lie | 0.1566 | 0.2029 | 0.1641 | 0.0151 |          0.1619 |          0.1643 |              0.0647 |    0.4937 |
| + BERT + Power | suspected_lie | 0.1243 | 0.2023 | 0.1454 | 0.018  |          0.1424 |          0.1456 |              0.0601 |    0.4743 |

The test positive prevalences are 8.76% (actual) and 6.67% (suspected). BERT-only test ranking is approximately random for actual lie (ROC-AUC 0.5100) and below random for suspected lie (0.4937). BERT+Power gives ROC-AUC 0.5599 (actual) and 0.4743 (suspected). PR-AUC/average precision is also close to the population positive prevalence. Validation ranking may differ from test; the available validation split has only **one game**.

## Epoch-level evidence

| architecture      | task          |   epochs_logged |   best_epoch |   first_training_loss |   last_training_loss |   first_validation_loss |   last_validation_loss |   max_validation_lie_f1 |
|:------------------|:--------------|----------------:|-------------:|----------------------:|---------------------:|------------------------:|-----------------------:|------------------------:|
| Hierarchical LSTM | actual_lie    |              10 |            5 |                0.6098 |               0.0753 |                  0.7082 |                 4.5195 |                  0.104  |
| Hierarchical LSTM | suspected_lie |              15 |           11 |                0.6754 |               0.083  |                  0.7895 |                 3.7442 |                  0.1538 |
| + BERT            | actual_lie    |               6 |            1 |                0.4927 |               0.4794 |                  0.7245 |                 0.8224 |                  0      |
| + BERT            | suspected_lie |               6 |            1 |                0.5076 |               0.496  |                  0.7612 |                 0.7637 |                  0      |
| + BERT + Power    | actual_lie    |               6 |            1 |                0.4911 |               0.481  |                  0.8645 |                 0.8218 |                  0      |
| + BERT + Power    | suspected_lie |               6 |            1 |                0.5112 |               0.493  |                  0.7901 |                 0.8428 |                  0      |
| + Power           | actual_lie    |               9 |            4 |                0.6352 |               0.0865 |                  0.8287 |                 3.2691 |                  0.1263 |
| + Power           | suspected_lie |              12 |            7 |                0.6314 |               0.0852 |                  0.6962 |                 3.9173 |                  0.1212 |

The four BERT variants selected **epoch 1**, with zero validation positive-class F1 throughout the recorded epochs. The training losses are finite and vary; this is insufficient to prove whether BERT received correct gradients, whether layer outputs collapsed, or which mechanism caused the failed classifications. The selected architectures use `bert-base-uncased` pooler output with configured `freeze: false`, 20 tokens and 8-message conversation chunks. Configuration alone cannot verify effective fine-tuning.

## Exploratory threshold analysis (validation-selected only)

Candidate thresholds were fixed to 0.01–0.50 inclusive, in 0.01 increments. For each model/task, the positive-class F1-maximizing threshold was selected using the **validation split only** (ties favor the higher threshold). These fixed thresholds were then applied once to test predictions. This is *supplementary analysis* and does not replace the original fixed-threshold results.

| architecture   | task          |   validation_selected_threshold | test_original_positive_f1   | test_adjusted_positive_f1   | test_adjusted_macro_f1   | test_adjusted_precision   | test_adjusted_recall   |   test_adjusted_predicted_positive |
|:---------------|:--------------|--------------------------------:|:----------------------------|:----------------------------|:-------------------------|:--------------------------|:-----------------------|-----------------------------------:|
| + BERT         | actual_lie    |                            0.22 | 0.00%                       | 12.93%                      | 47.59%                   | 8.72%                     | 25.00%                 |                                688 |
| + BERT + Power | actual_lie    |                            0.17 | 0.00%                       | 11.45%                      | 51.05%                   | 10.56%                    | 12.50%                 |                                284 |
| + BERT         | suspected_lie |                            0.2  | 0.00%                       | 4.59%                       | 47.19%                   | 3.50%                     | 6.67%                  |                                314 |
| + BERT + Power | suspected_lie |                            0.18 | 0.00%                       | 4.64%                       | 47.27%                   | 3.56%                     | 6.67%                  |                                309 |

A lower threshold recovers some actual-lie positives (BERT-only test lie F1 12.93%; BERT+Power 11.45%), but at low precision. For suspected lie, adjusted F1 remains below 5%. Thus threshold choice explains the zero *binary* positive predictions, but **not** satisfactory discrimination. Validation-selected thresholds are highly unstable with one validation game; no threshold was tuned on test.

## What is and is not established

**Confirmed:** exact majority-only test and validation decisions at default threshold; narrow, nonconstant predicted-score distributions; weak test class separation; six recorded training epochs per BERT variant with epoch-1 selection and zero validation positive F1; correct prediction/metric alignment.

**Plausible but unproven:** inadequate optimization under class imbalance, ineffective representation learning, insufficient training budget or domain mismatch. Gradient norms, message-embedding variance, optimizer trajectories, and BERT layer gradients were **not** supplied, so causal diagnosis is incomplete. Avoid claiming that a specific bug, attention pattern, or weight setting is the cause.

## Reproduction

Run `python -m diplomacy.analysis.phase6_phase5_deep_dive --root .`. See `artifacts/phase_6/bert_diagnostics/` and `artifacts/phase_6/plots/` for exact computed tables and score plots.
