# Phase 6 — Interpretability and uncertainty (evidence-aware update)

> **Final verification (2026-11-08):** The initial lightweight archive did not include Phase 5 predictions or epoch histories. A later, separately supplied `phase6-required-analysis-files.zip` resolved both gaps. In the assembled Phase 1–6 working copy, all 16 Phase 5 prediction CSVs and eight `epochs.jsonl` logs are present. The earlier "missing" statements below are preserved as historical provenance, **not the current artifact status**. See `docs/phase_6_final_verification.md` and `artifacts/phase_6/audit/final_verification.json`. No model retraining, GPU run or fresh checkpoint inference was performed during this final verification.

## What the provided files support

The sixteen Phase 5 prediction CSVs support probability histograms, precision-recall ranking summaries, ROC-AUC, Brier scores, confidence and disagreement analyses. All Phase 5 predictions were aligned by stable message identifiers. Detailed tables live under `artifacts/phase_6/bert_diagnostics/` and `artifacts/phase_6/error_analysis/`.

The BERT score distributions have limited variation, with ROC-AUC around 0.5 for the BERT-only models on both test tasks. A validation-selected threshold recovers some positive predictions but leaves suspected-lie test positive F1 under 5%. These findings are predictive diagnostics, not evidence about what a particular BERT attention head or representation encoded.

## What is not identified

No input gradient, feature-attribution map, context-window counterfactual, internal embedding tensor or per-message score-delta feature was supplied. Consequently this report does not claim feature causality, the effect of modifying a score delta while holding all else fixed, or an explanation for any individual player's intentions. To test feature reliance, use independently defined perturbations of the same inputs and preserve the original baseline predictions.

Calibration diagnostics, where computed, are descriptive only: the test split contains just two games and a small minority positive population. Brier scores in `probability_distributions.csv` are not evidence of a well-calibrated deception detector by themselves.
