# Phase 5 — Hierarchical LSTM, BERT and Power Dynamics

**Status (8 November 2026):** All eight Phase 5 model/task configurations were trained in Google Colab and evaluated on the held-out test data. On Windows CPU, all eight saved Colab parity checks passed (two reference cases per experiment; 16/16 comparisons), and the full Phase 1–5 suite passed **62/62 tests** (Python 3.14.7). The local parity checks used inference-only checkpoints. The user-supplied `test_metrics.json` console output is the numerical source for the following eight test records. Training histories and validation curves have not been independently supplied here; best epochs are read from the test metrics metadata.

## 1. Research purpose and experimental design

Study the contributions of **relative power** (`game_score_delta`: sender score minus recipient score) and **BERT-based message representations** to a forward conversation-level Hierarchical LSTM. All models are trained separately for `actual_lie` and `suspected_lie` using original game-level splits. Label `1` means actual/suspected lie; missing receiver annotations are retained as context but masked out of supervised loss and evaluation.

| Model ID | Message encoder | Context encoder | Additional input |
|---|---|---|---|
| `hierarchical_lstm` | Phase 4 GloVe BiLSTM + pooling | Forward LSTM | None |
| `hierarchical_lstm_power` | Same as baseline | Forward LSTM | Train-normalized power differential |
| `hierarchical_lstm_bert` | Fine-tuned `bert-base-uncased` pooler | Forward 200-hidden LSTM | None |
| `hierarchical_lstm_bert_power` | Same BERT pooler | Forward 200-hidden LSTM | Train-normalized power differential |

Power-feature preprocessing parameters are fitted on the training split and checkpointed. The model architecture and score mapping are described in [Phase 5 research design](phase_5_research_design.md). The BERT variants are a **PyTorch approximation** of the paper's historical AllenNLP setup, and the non-BERT baseline uses transferred Phase 4 initialization. Context chunks reset at bounded lengths (8 messages by default for Phase 5) instead of carrying full-game history. These are methodological differences, not exact reproduction claims.

## 2. Final held-out test results

### Actual lie detection

**Test support:** 2,741 messages; 2,501 truthful and 240 lies (positive prevalence 8.76%). All four use the same test count.

| Architecture | Accuracy | Macro F1 | Positive F1 | Positive precision | Positive recall | PR-AUC | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hierarchical LSTM | 89.31% | 52.63% | 10.94% | 20.22% | 7.50% | 0.1341 | 5 |
| Hierarchical LSTM + Power | 85.26% | 57.95% | 24.06% | 21.92% | 26.67% | 0.1691 | 4 |
| Hierarchical LSTM + BERT | 91.24% | 47.71% | 0.00% | 0.00% | 0.00% | 0.0900 | 1 |
| Hierarchical LSTM + BERT + Power | 91.24% | 47.71% | 0.00% | 0.00% | 0.00% | 0.1064 | 1 |

### Suspected lie detection

**Test support:** 2,475 annotated messages; 2,310 not suspected and 165 suspected lies (positive prevalence 6.67%). Messages with missing receiver annotations are excluded from evaluation.

| Architecture | Accuracy | Macro F1 | Positive F1 | Positive precision | Positive recall | PR-AUC | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hierarchical LSTM | 90.30% | 50.93% | 6.98% | 9.68% | 5.45% | 0.0890 | 11 |
| Hierarchical LSTM + Power | 87.84% | 53.36% | 13.26% | 12.64% | 13.94% | 0.0981 | 7 |
| Hierarchical LSTM + BERT | 93.33% | 48.28% | 0.00% | 0.00% | 0.00% | 0.0647 | 1 |
| Hierarchical LSTM + BERT + Power | 93.33% | 48.28% | 0.00% | 0.00% | 0.00% | 0.0601 | 1 |

**Metric definitions:** Positive precision, recall, and F1 refer to class 1 (lie/suspected lie); Macro F1 is the unweighted mean of class 0 and class 1 F1 scores. Accuracy is potentially misleading with strong class imbalance. PR-AUC is reported as supplied by the experiment metrics, without claiming a new significance test.

## 3. Confusion matrices

Rows are true labels and columns are predicted labels: `[[TN, FP], [FN, TP]]`.

### Actual lie

| Architecture | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Hierarchical LSTM | 2430 | 71 | 222 | 18 |
| Hierarchical LSTM + Power | 2273 | 228 | 176 | 64 |
| Hierarchical LSTM + BERT | 2501 | 0 | 240 | 0 |
| Hierarchical LSTM + BERT + Power | 2501 | 0 | 240 | 0 |

### Suspected lie

| Architecture | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Hierarchical LSTM | 2226 | 84 | 156 | 9 |
| Hierarchical LSTM + Power | 2151 | 159 | 142 | 23 |
| Hierarchical LSTM + BERT | 2310 | 0 | 165 | 0 |
| Hierarchical LSTM + BERT + Power | 2310 | 0 | 165 | 0 |

The BERT-only and BERT+Power models predicted **every test sample as class 0**, on both tasks. Consequently, lie recall and lie F1 are zero. Their higher accuracies (91.24% actual; 93.33% suspected) are simply majority-class accuracies, **not successful lie detection**. The reported PR-AUCs are nonzero because their underlying ranking probabilities can still vary despite the all-negative thresholded predictions.

## 4. Ablation comparisons within Phase 5

### Actual-lie changes relative to Phase 5 Hierarchical LSTM

| Variant versus baseline | Δ Macro F1 | Δ positive F1 | Δ accuracy |
|---|---:|---:|---:|
| Hierarchical LSTM + Power | +5.32 pp | +13.12 pp | -4.05 pp |
| Hierarchical LSTM + BERT | -4.92 pp | -10.94 pp | +1.93 pp |
| Hierarchical LSTM + BERT + Power | -4.92 pp | -10.94 pp | +1.93 pp |

### Suspected-lie changes relative to Phase 5 Hierarchical LSTM

| Variant versus baseline | Δ Macro F1 | Δ positive F1 | Δ accuracy |
|---|---:|---:|---:|
| Hierarchical LSTM + Power | +2.43 pp | +6.28 pp | -2.46 pp |
| Hierarchical LSTM + BERT | -2.65 pp | -6.98 pp | +3.03 pp |
| Hierarchical LSTM + BERT + Power | -2.65 pp | -6.98 pp | +3.03 pp |

**Power:** The +Power model has the highest Phase 5 Macro F1 and minority-class F1 on **both** tasks. For actual lies, its lie F1 rises from 10.94% to 24.06%, with recall from 7.50% to 26.67%. This improvement comes with more false positives and lower overall accuracy. For suspected lies, its positive F1 rises from 6.98% to 13.26%.

**BERT:** Both BERT variants collapsed to majority-class predictions at the decision threshold selected in these runs. Adding Power to BERT did not recover positive predictions; the two BERT variants have identical confusion matrices and classification F1 within each task, although their PR-AUCs differ. This supports a conclusion about **these configurations**, not about BERT in general. Potential causes such as threshold choice, class imbalance, learning-rate dynamics, early stopping, and domain mismatch require follow-up experiments before attributing causality.

## 5. Cross-phase held-out comparison

The earlier rows are from previously recorded Phase 2–4 results in the project documentation; Phase 5 rows come from the eight supplied test metric records. For detailed model-wise comparison and provenance, see [Model comparison](phase_5_results_comparison.md).

| Phase and model | Actual Macro F1 | Actual positive F1 | Suspected Macro F1 | Suspected positive F1 |
|---|---:|---:|---:|---:|
| Phase 2 — BoW + Power | 54.72% | 22.88% | 51.13% | 6.87% |
| Phase 3 — Standard LSTM | 53.12% | 14.19% | 51.70% | 9.79% |
| Phase 4 — BiLSTM | 54.79% | 16.55% | 51.77% | 8.26% |
| Phase 4 — Concatenated Context | 52.72% | 12.21% | 51.35% | 8.89% |
| Phase 4 — Hierarchical Context LSTM | 53.32% | 12.78% | 48.54% | 1.05% |
| **Phase 5 — Hierarchical LSTM** | 52.63% | 10.94% | 50.93% | 6.98% |
| **Phase 5 — Hierarchical LSTM + Power** | **57.95%** | **24.06%** | **53.36%** | **13.26%** |
| Phase 5 — Hierarchical LSTM + BERT | 47.71% | 0.00% | 48.28% | 0.00% |
| Phase 5 — Hierarchical LSTM + BERT + Power | 47.71% | 0.00% | 48.28% | 0.00% |

Among the **listed** results, Phase 5 +Power is best on Macro F1 and positive F1 for both tasks; this does not necessarily establish superiority over every traditional baseline or over repeat-run uncertainty. Phase 5 and Phase 4 hierarchical variants are **not identical runs**: they differ in initialization and context settings. No confidence intervals or multi-seed statistical tests are available.

## 6. Training, checkpointing and portability

- Colab notebook: `notebooks/13_phase_5_hierarchical_lstm_bert_power_colab.ipynb`.
- Per-experiment YAML configurations: `configs/experiments/neural/phase_5/{architecture}/{task}.yaml`.
- Training/validation/test and inference: `src/diplomacy/training/phase5_runner.py`.
- Best and resume checkpoints (original copies kept in Drive): `artifacts/phase_5/checkpoints/{architecture}/{task}/best_model.pt` and `latest_checkpoint.pt`.
- Final test metrics: `artifacts/phase_5/metrics/{architecture}/{task}/test_metrics.json`.
- Local parity references: `artifacts/phase_5/checkpoints/{architecture}/{task}/parity_reference.json`.
- Local BERT tokenizer directories: `artifacts/phase_5/bert_tokenizer/hierarchical_lstm_bert/` and `artifacts/phase_5/bert_tokenizer/hierarchical_lstm_bert_power/`.

The local `best_model.pt` files can be **inference-only checkpoints** stripped of optimizer/scheduler states. This is appropriate for predictions and parity but **not** for faithfully resuming training. Retain the full Colab Drive backups for training resume. Local `transformers>=4.40,<5` is required for BERT variants. The checkpoint loader must reconstruct `AutoConfig.for_model(...)` without passing `model_type` twice; this compatibility fix was applied locally and in Colab. The existing Phase 4 context truncation fix does **not** retroactively update Phase 4 trained checkpoints or historical evaluation scores.

### Windows PowerShell verification

```powershell
$env:PYTHONPATH = 'src'
python -m pytest tests/ -v

foreach ($arch in 'hierarchical_lstm','hierarchical_lstm_power','hierarchical_lstm_bert','hierarchical_lstm_bert_power') {
    foreach ($task in 'actual_lie','suspected_lie') {
        python -m diplomacy.training.phase5_runner --root . --architecture $arch --task $task --action parity --device cpu
        if ($LASTEXITCODE -ne 0) { throw "Parity failed: $arch / $task" }
    }
}
```

**Observed on Windows:** 62 tests passed; eight parity invocations returned `passed=True`, each checking two saved Colab reference cases (16 comparisons total). No new CPU predictions are claimed to have been executed by this documentation-writing step.

## 7. Limitations and scientific interpretation

1. Dataset has only 12 games, and the test partition contains 2 games; uncertainty across games may be substantial.
2. Lie/suspected-lie prevalence is low, making accuracy especially easy to inflate by predicting only class 0.
3. The Phase 5 non-BERT baseline reuses Phase 4 model components/initialization and is not an independently initialized full-history reproduction of the original context LSTM.
4. Conversation context is chunked for free-tier GPU constraints, which truncates history.
5. BERT message inputs are capped at 20 pieces by the recovered original configuration; long-message evidence may be truncated.
6. Results reflect one experimental setup, not a controlled multi-seed ablation. No statistical significance claim is made.
7. Fine-tuned BERT variants displayed all-negative decisions at the saved threshold; threshold diagnostics and error analysis are warranted, but no test-driven threshold retuning should be used to inflate these results.
8. The source-provided metrics are the primary evidence here. Full individual prediction files, calibration plots, training histories, and hardware timing were not inspected in this documentation generation step.

## 8. Reproducibility checks and status

| Item | Status |
|---|---|
| Eight Phase 5 experiment configurations | Implemented and trained in Colab |
| Held-out test metrics | Eight JSON results supplied and documented |
| Windows Phase 1–5 unit tests | 62/62 passed |
| Colab-to-Windows CPU prediction parity | 8/8 experiments; 16 saved reference cases passed |
| Windows BERT tokenizer and model reconstruction | Verified indirectly by successful local parity |
| Original `data/` game-level splits | Preserved by pipeline and unit tests |
| Full training-state checkpoints in Drive | Retained by user; not part of lightweight download |

**Current status:** Phase 5 experiments and local portability verification are complete. The recommended next research work is multi-seed uncertainty estimation, class-imbalance diagnostics, and validation-only calibration experiments; these are future work, not results reported above.
