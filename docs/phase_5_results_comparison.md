# Held-out test comparison — Phases 2–5

This document distinguishes **Phase 5 values extracted from the user-provided eight `test_metrics.json` records** from **historical Phase 2–4 values in the completed project documentation**. All values are test-set percentages, except raw confusion matrix counts. For the earlier phases, the underlying JSON files were not provided with this documentation request; verify against your local `artifacts/` when making a publication-ready table.

**Actual lie:**

| Architecture | Accuracy | Macro F1 | Positive F1 | Positive precision | Positive recall | PR-AUC | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hierarchical LSTM | 89.31% | 52.63% | 10.94% | 20.22% | 7.50% | 0.1341 | 5 |
| Hierarchical LSTM + Power | 85.26% | 57.95% | 24.06% | 21.92% | 26.67% | 0.1691 | 4 |
| Hierarchical LSTM + BERT | 91.24% | 47.71% | 0.00% | 0.00% | 0.00% | 0.0900 | 1 |
| Hierarchical LSTM + BERT + Power | 91.24% | 47.71% | 0.00% | 0.00% | 0.00% | 0.1064 | 1 |

**Suspected lie:**

| Architecture | Accuracy | Macro F1 | Positive F1 | Positive precision | Positive recall | PR-AUC | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hierarchical LSTM | 90.30% | 50.93% | 6.98% | 9.68% | 5.45% | 0.0890 | 11 |
| Hierarchical LSTM + Power | 87.84% | 53.36% | 13.26% | 12.64% | 13.94% | 0.0981 | 7 |
| Hierarchical LSTM + BERT | 93.33% | 48.28% | 0.00% | 0.00% | 0.00% | 0.0647 | 1 |
| Hierarchical LSTM + BERT + Power | 93.33% | 48.28% | 0.00% | 0.00% | 0.00% | 0.0601 | 1 |

## Historical context and Phase 5 comparison

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


## Confusion matrices

Actual lie:

| Architecture | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Hierarchical LSTM | 2430 | 71 | 222 | 18 |
| Hierarchical LSTM + Power | 2273 | 228 | 176 | 64 |
| Hierarchical LSTM + BERT | 2501 | 0 | 240 | 0 |
| Hierarchical LSTM + BERT + Power | 2501 | 0 | 240 | 0 |

Suspected lie:

| Architecture | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Hierarchical LSTM | 2226 | 84 | 156 | 9 |
| Hierarchical LSTM + Power | 2151 | 159 | 142 | 23 |
| Hierarchical LSTM + BERT | 2310 | 0 | 165 | 0 |
| Hierarchical LSTM + BERT + Power | 2310 | 0 | 165 | 0 |

## Key conclusion

The Phase 5 +Power variant is the highest-performing of the four Phase 5 variants on Macro F1 and positive-class F1 for both tasks. Both BERT variants predict only class 0 at the saved decision threshold. There is no multi-seed confidence interval or significance test, so results should be interpreted as observed differences in these runs.
