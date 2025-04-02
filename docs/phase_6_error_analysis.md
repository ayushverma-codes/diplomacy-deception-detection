# Phase 6 — Phase 5 message-level error and game analysis

> **Final verification (2026-11-08):** The initial lightweight archive did not include Phase 5 predictions or epoch histories. A later, separately supplied `phase6-required-analysis-files.zip` resolved both gaps. In the assembled Phase 1–6 working copy, all 16 Phase 5 prediction CSVs and eight `epochs.jsonl` logs are present. The earlier "missing" statements below are preserved as historical provenance, **not the current artifact status**. See `docs/phase_6_final_verification.md` and `artifacts/phase_6/audit/final_verification.json`. No model retraining, GPU run or fresh checkpoint inference was performed during this final verification.

## Verified dataset-aligned outputs

For each test message the original Phase 5 prediction files contain `game_id`, `message_id`, `absolute_message_index`, `target`, `prediction`, and `prob_lie`. Every model/task population matched the baseline exactly. Generated outputs include:

- `artifacts/phase_6/error_analysis/phase5_all_errors.csv`: false positives and false negatives with original message identifiers and confidence; no fabricated message texts.
- `artifacts/phase_6/error_analysis/phase5_disagreements_vs_baseline.csv`: cases with differing class predictions against the Phase 5 hierarchical baseline.
- `artifacts/phase_6/error_analysis/phase5_per_game_metrics.csv`: separate game-level results.

## Per-game results, at original decision threshold

| task          | architecture      |   game_id |   samples |   positives | lie_f1   | lie_recall   |
|:--------------|:------------------|----------:|----------:|------------:|:---------|:-------------|
| actual_lie    | Hierarchical LSTM |        12 |      1195 |          47 | 7.32%    | 6.38%        |
| actual_lie    | Hierarchical LSTM |         4 |      1546 |         193 | 12.15%   | 7.77%        |
| actual_lie    | + Power           |        12 |      1195 |          47 | 13.70%   | 21.28%       |
| actual_lie    | + Power           |         4 |      1546 |         193 | 27.98%   | 27.98%       |
| actual_lie    | + BERT            |        12 |      1195 |          47 | 0.00%    | 0.00%        |
| actual_lie    | + BERT            |         4 |      1546 |         193 | 0.00%    | 0.00%        |
| actual_lie    | + BERT + Power    |        12 |      1195 |          47 | 0.00%    | 0.00%        |
| actual_lie    | + BERT + Power    |         4 |      1546 |         193 | 0.00%    | 0.00%        |
| suspected_lie | Hierarchical LSTM |        12 |      1111 |          25 | 0.00%    | 0.00%        |
| suspected_lie | Hierarchical LSTM |         4 |      1364 |         140 | 9.00%    | 6.43%        |
| suspected_lie | + Power           |        12 |      1111 |          25 | 2.08%    | 4.00%        |
| suspected_lie | + Power           |         4 |      1364 |         140 | 17.53%   | 15.71%       |
| suspected_lie | + BERT            |        12 |      1111 |          25 | 0.00%    | 0.00%        |
| suspected_lie | + BERT            |         4 |      1364 |         140 | 0.00%    | 0.00%        |
| suspected_lie | + BERT + Power    |        12 |      1111 |          25 | 0.00%    | 0.00%        |
| suspected_lie | + BERT + Power    |         4 |      1364 |         140 | 0.00%    | 0.00%        |

**Caution:** Game 4 contains 193 of the 240 actual lies and 140 of the 165 suspected lies; game 12 contains 47 and 25 respectively. All pooled metrics are heavily influenced by Game 4. Differences in positive-class F1 across the two games demonstrate instability; message counts are **not** independent replicates. No confidence intervals or significance claims are made.

## Scope of interpretation

The CSVs do not contain original message text, sender and receiver names, season, power feature values, or surrounding messages. Message-level labels, confidence, and disagreements can be checked rigorously; natural-language descriptions of error examples or qualitative claims about speaker intentions would require joining the original JSONL using stable IDs. That enrichment has not been performed in this update.

At default threshold BERT false positives are **zero**, while BERT false negatives include **all** 240 actual lies and **all** 165 suspected lies on the held-out split for both BERT model variants.
