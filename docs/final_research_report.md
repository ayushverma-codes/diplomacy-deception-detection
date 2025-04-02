# Diplomacy Deception Detection — Research report (Phases 1–6)

> **Final verification (2026-11-08):** The initial lightweight archive did not include Phase 5 predictions or epoch histories. A later, separately supplied `phase6-required-analysis-files.zip` resolved both gaps. In the assembled Phase 1–6 working copy, all 16 Phase 5 prediction CSVs and eight `epochs.jsonl` logs are present. The earlier "missing" statements below are preserved as historical provenance, **not the current artifact status**. See `docs/phase_6_final_verification.md` and `artifacts/phase_6/audit/final_verification.json`. No model retraining, GPU run or fresh checkpoint inference was performed during this final verification.

## Research question

Reproduce and analyze deception detection in negotiation-oriented Diplomacy conversations, distinguishing *actual lies* (sender-labelled) from *suspected lies* (receiver-labelled). Original study: Peskov et al., ACL 2020, https://aclanthology.org/2020.acl-main.353/.

## Dataset and design

Original dataset: 17,289 messages across 12 games, preserved as game-level train/validation/test splits. The held-out test comprises two games; 2,741 actual-lie examples (240 positive) and 2,475 annotated suspected-lie examples (165 positive). Unannotated receiver labels are excluded from supervised suspected-lie evaluation.

## Implemented research progression

Phase 1: ingestion, validation, preprocessing and EDA. Phase 2: traditional machine learning and Power. Phase 3: standard message LSTM. Phase 4: BiLSTM, concatenated context and hierarchical context. Phase 5: hierarchical baseline, +Power, +BERT and +Power+BERT. Phase 6: artifact audit, retrospective ablation, score summaries, partial error analysis and benchmarking.

## Verified Phase 5 test results (from supplied test metric JSON)

| architecture                 | task          | accuracy   | macro_f1   | lie_f1   | precision   | recall   |
|:-----------------------------|:--------------|:-----------|:-----------|:---------|:------------|:---------|
| hierarchical_lstm            | actual_lie    | 89.31%     | 52.63%     | 10.94%   | 20.22%      | 7.50%    |
| hierarchical_lstm            | suspected_lie | 90.30%     | 50.93%     | 6.98%    | 9.68%       | 5.45%    |
| hierarchical_lstm_power      | actual_lie    | 85.26%     | 57.95%     | 24.06%   | 21.92%      | 26.67%   |
| hierarchical_lstm_power      | suspected_lie | 87.84%     | 53.36%     | 13.26%   | 12.64%      | 13.94%   |
| hierarchical_lstm_bert       | actual_lie    | 91.24%     | 47.71%     | 0.00%    | 0.00%       | 0.00%    |
| hierarchical_lstm_bert       | suspected_lie | 93.33%     | 48.28%     | 0.00%    | 0.00%       | 0.00%    |
| hierarchical_lstm_bert_power | actual_lie    | 91.24%     | 47.71%     | 0.00%    | 0.00%       | 0.00%    |
| hierarchical_lstm_bert_power | suspected_lie | 93.33%     | 48.28%     | 0.00%    | 0.00%       | 0.00%    |

## Main findings

**Power improves the measured minority-class F1** in the non-BERT hierarchy: actual lie 10.94%→24.06%, suspected lie 6.98%→13.26%. The gain comes with more false positives. These are observed associations across independently trained models, not causal claims.

**The BERT variants fail to flag any positive test examples** at their default threshold. Their accuracy (91.24% actual, 93.33% suspected) is explained by majority-class prevalence; zero lie recall reveals the failure. Precise causes remain unknown without training histories, validation probabilities and/or gradient/representation diagnostics.

**Context models do not consistently dominate message-only baselines.** Phase 4 per-game results differ, and controlled context removal was not conducted. Phase 5 +Power is the strongest tested Phase 5 model on both tasks by Macro F1 and lie F1, based on supplied metric JSON.

## Historical comparison

The original paper reports actual-lie Context LSTM+Power Macro F1 57.2 and Lie F1 27.0; suspected-lie 53.3 and 13.0. Our implementation's corresponding Phase 5 reported numbers are 57.95/24.06 and 53.36/13.26. These should not be treated as exact reproduction without matching architecture and experimental protocols.

## Research limitations

- Only 12 games (two test games); messages are not independent data points.
- Phase 5 prediction CSVs and training histories not present in the shared export.
- The Phase 6 analysis did not rerun models, probe gradients or perform threshold optimization.
- Model checkpoint loading and CPU parity previously succeeded locally, but that observation is user-reported rather than verified by the present archive.
- Original dataset and source code were not bundled with the Phase 6 analysis export, limiting textual error examples and source-level diagnosis.

## Research follow-up

Obtain Phase 5 **validation and test prediction CSVs**, then validate game/message alignment and explore validation-only thresholds for BERT. Obtain histories, then inspect loss and gradient flow. Preserve original test metrics and checkpoints as immutable baselines; label any new thresholded evaluation as supplementary. New Colab training is not warranted by current evidence alone.

## Supporting reports

- `docs/phase_6_initial_audit.md`
- `docs/phase_6_bert_failure_analysis.md`
- `docs/phase_6_ablation_study.md`
- `docs/phase_6_error_analysis.md`
- `docs/phase_6_model_interpretability.md`
- `docs/phase_6_research_validation.md`

**Status:** Phase 6 artifact-based analysis completed to the extent the provided files permit. Full Phase 5 message-level and training-diagnostic stages remain pending.


**Scope and provenance.** Source: user-supplied `phase6-analysis-lightweight.zip` (154 files); derived outputs from `run_phase6_analysis.py`. The original Phase 1–5 project code and model weights were not part of this archive. The eight Phase 5 metric JSON files were checked for internal arithmetic consistency against their confusion matrices, not against absent Phase 5 message-level predictions.


---

# Phase 6 follow-up — Verified Phase 5 predictions and learning diagnostics

The initial reports were built without the Phase 5 prediction CSVs. A subsequent separate upload supplied all 16 Phase 5 validation/test prediction files, eight epoch logs, eight summaries, and matching metric JSON files. The new analysis **recomputed and verified** all prediction counts, F1 values, confusion matrices, and model-to-model target/message alignment. It also generated threshold, per-game, probability distribution, and disagreement data. Earlier text describing these Phase 5 artifacts as missing is superseded by this section.

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

Main finding: Phase 5 Hierarchical LSTM + Power remains the best of the four Phase 5 variants under the original held-out test and threshold (Macro F1 57.95% actual / 53.36% suspected). BERT-only and BERT+Power predict only the negative class at the original threshold. Their score distributions are not constant, but test ROC-AUC is weak (approximately 0.47–0.56); class separation is inadequate. All BERT training runs selected epoch 1 and recorded zero validation lie F1 across their epochs. The available evidence does not identify a particular root-cause bug.

A validation-only threshold sweep recovers actual-lie positives for BERT-only (test positive F1 12.93%) and BERT+Power (11.45%), but suspected-lie positive F1 remains below 5% and precision is low. This is an exploratory supplement, **not** a replacement for the original published/project scores. The test split contains only games 4 and 12; no statistical significance or causal claims are made.

See updated `phase_6_bert_failure_analysis.md`, `phase_6_ablation_study.md`, and `phase_6_error_analysis.md` for the complete evidence.


## Phase 6 final component status

The final verification completed 16 Phase 5 split-metric recomputations, all four-architecture alignment checks, validation-selected threshold analysis, BERT probability and training-log diagnostics, per-game error breakdowns, Power/BERT ablations, and Phase 2–5 population alignment. The full assembled project suite passed **71 tests**; duplicate Phase 4 context function removed with no behavioral changes. **No GPU training or fresh checkpoint parity run** was conducted in this verification. The originally missing Phase 5 prediction CSVs and epoch histories were later supplied and are now verified; earlier statements remain as provenance, not current status. The final report and generated analysis are reproducible from the assembled project **with all research artifacts** but not source alone. See `docs/phase_6_final_verification.md` and `artifacts/phase_6/audit/final_verification.json`.
