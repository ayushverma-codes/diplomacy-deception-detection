# Phase 1 data analysis findings

See `data/reports/dataset_statistics.json`, `data/reports/validation_report.json`, `data/reports/data_summary.md` for audited numeric summaries. Five analysis notebooks are located in `notebooks/`.

The 17,289 observed messages consist of 13,132 training, 1,416 validation, and 2,741 test messages. These occur in 9, 1, and 2 distinct games respectively, matching the original paper's experimental split. The raw JSONL rows represent two-person conversation segments, not individual games.

The source truth-label convention is reverse to the usual positive-class convention (`False` means deception); receiver `NOANNOTATION` indicates a genuine missing label. The paper reports strong class imbalance. Notebook 03 quantifies this with annotated-message denominators separately for the two tasks. Differences between textual features or score differentials and lie labels are descriptive correlations only. Real-world extrapolation, receiver annotation missingness, class imbalance and game/player-specific language are limitations.

No classification model has been trained, and no significance or causal claims are made from EDA.

## Audited task counts

| Split | Actual lies / valid | Suspected lies / annotated | Missing receiver labels |
|---|---:|---:|---:|
| train | 591 / 13,132 | 566 / 12,025 | 1,107 |
| validation | 56 / 1,416 | 47 / 1,283 | 133 |
| test | 240 / 2,741 | 165 / 2,475 | 266 |

There are **887 actual lies** in 17,289 messages, **778 suspected lies** in 15,783 receiver-annotated messages, and **1,506 missing receiver annotations**. Six original conversation records are empty. The audit found no array-alignment problems or duplicate `(game_id, absolute_message_index)` pairs. The notebook results demonstrate descriptive distributions, not evidence of causal detection signals.
