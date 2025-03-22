# Phase 2: Traditional Machine Learning Baselines

Paper: Peskov et al. (ACL 2020), *It Takes Two to Lie: One to Lie, and One to Listen*. Original paper: https://aclanthology.org/2020.acl-main.353/ . Original authors' public repository: https://github.com/DenisPeskoff/2020_acl_diplomacy .

## Inspection and reuse

Phase 1 loaders, `parser.py`, `conversation_builder.py`, `preprocessing.py`, and `validator.py` remain reusable; the Phase 1 pipeline's generated output paths were moved from `data/{processed,reports}` to `artifacts/{processed,reports_phase1}`. The original three JSONL files remain byte-unchanged in `data/`, and duplicate `data/raw/` copies were removed after byte comparison. All existing notebooks 01–06 remain unchanged.

Raw records are **two-player conversations** with parallel arrays for message text, speakers, receivers, two kinds of labels, game indices, seasons, years, scores and score deltas. Processed records are message-level JSONL, retaining `message_id`, ordering metadata, raw labels, message text and `target` with 1=lie. `NOANNOTATION` is dropped only from suspected-lie data.

## Experimental methods

All ten requested experiment IDs were trained/evaluated for each task. Random predictor samples from the training-set class prior using a fixed seed. Majority predictor uses training majority only. Other experiments use `sklearn.linear_model.LogisticRegression` (`liblinear`, max_iter=1500), hyperparameter grid `C ∈ {0.1, 1, 10}`, `class_weight ∈ {None, 'balanced'}`; select the top validation Macro F1 and break ties by positive-class F1; report test metrics once under the selected setting. BoW `CountVectorizer` fits training messages only; the vocabulary is reused for validation/test. Length statistics are message-internal prediction-time features and exclude game scores. Power uses the original numeric `game_score_delta` which Phase 1 names `score_delta`, with training-only median imputation/standardization. Sparse matrices are used for BoW and combined features. Evaluation reports accuracy, precision, recall, F1, Macro F1, positive-class F1, confusion matrix, and average precision when supported. Results, predictions with message IDs, fitted models and feature extractors are written under `artifacts/`.

## Paper reference versus reproduced tests

All values below are **percent F1 on the test split**. Model numbering follows the prompt, not the original paper. No test-set tuning was performed.

### actual_lie

| Model | Validation Macro F1 | Test Macro F1 | Test Lie F1 | Published Macro F1 | Published Lie F1 |
|---|---:|---:|---:|---:|---:|
| 01_random | 49.53 | 49.68 | 5.75 | 39.8 | 14.9 |
| 02_majority | 48.99 | 47.71 | 0.00 | 47.8 | 0.0 |
| 03_harbingers | 48.99 | 47.71 | 0.00 | 52.8 | 24.6 |
| 04_logistic_regression | 49.32 | 50.49 | 21.92 | — | — |
| 05_harbingers_lr | 48.99 | 47.71 | 0.00 | — | — |
| 06_harbingers_power | 48.99 | 47.71 | 0.00 | 52.9 | 23.7 |
| 07_harbingers_lr_power | 48.99 | 47.71 | 0.00 | — | — |
| 08_bow_lr | 50.57 | 50.92 | 6.67 | 54.3 | 19.1 |
| 09_bow_power | 51.46 | 54.72 | 22.88 | 54.9 | 20.2 |
| 10_bow_lr_power | 51.46 | 54.72 | 22.88 | — | — |

- 03_harbingers: precision=0.000, recall=0.000, PR-AUC=0.141, confusion matrix=[[2501, 0], [240, 0]].

- 08_bow_lr: precision=0.300, recall=0.037, PR-AUC=0.143, confusion matrix=[[2480, 21], [231, 9]].

- 09_bow_power: precision=0.162, recall=0.388, PR-AUC=0.158, confusion matrix=[[2021, 480], [147, 93]].

### suspected_lie

| Model | Validation Macro F1 | Test Macro F1 | Test Lie F1 | Published Macro F1 | Published Lie F1 |
|---|---:|---:|---:|---:|---:|
| 01_random | 49.89 | 49.52 | 4.49 | 38.3 | 11.8 |
| 02_majority | 49.07 | 48.28 | 0.00 | 48.3 | 0.0 |
| 03_harbingers | 49.07 | 48.28 | 0.00 | 45.9 | 14.7 |
| 04_logistic_regression | 49.07 | 48.28 | 0.00 | — | — |
| 05_harbingers_lr | 49.07 | 48.28 | 0.00 | — | — |
| 06_harbingers_power | 49.07 | 48.28 | 0.00 | 45.1 | 15.5 |
| 07_harbingers_lr_power | 49.07 | 48.28 | 0.00 | — | — |
| 08_bow_lr | 51.30 | 50.66 | 5.98 | 51.5 | 13.7 |
| 09_bow_power | 51.41 | 51.13 | 6.87 | 51.6 | 13.9 |
| 10_bow_lr_power | 51.41 | 51.13 | 6.87 | — | — |

- 03_harbingers: precision=0.000, recall=0.000, PR-AUC=0.067, confusion matrix=[[2310, 0], [165, 0]].

- 08_bow_lr: precision=0.101, recall=0.042, PR-AUC=0.080, confusion matrix=[[2248, 62], [158, 7]].

- 09_bow_power: precision=0.118, recall=0.048, PR-AUC=0.081, confusion matrix=[[2250, 60], [157, 8]].

## Harbingers methodology and limitations

Table 8 in the original appendix lists eight categories: claim, subjectivity, expansion, contingency, premise, temporal future, temporal other, comparisons. The paper literally truncates the long **subjectivity** dictionary with an ellipsis. The feature extractor implements counts of boundary-matched category phrases listed in that table and only the visible subjectivity entries. **It is not an exact original lexicon.** Original code access was attempted, but the full lexicon could not be confirmed. Category counts can differ from the authors' model because their exact matching rules/classifier hyperparameters were not established. Therefore comparisons are indicative, not strict reproduction.

## BoW methodology

A training-only bag-of-words vocabulary is fitted with Unicode word-token matching; validation/test only transform, preventing vocabulary leakage. This differs from unspecified authors' exact tokenization, vocabulary pruning and weighting.

## Power methodology

The paper defines power differential as sender's victory-point count minus receiver's at message time (§4.2). The Phase 1 pipeline's `score_delta` is taken directly from source `game_score_delta` and the Phase 1 documentation identifies it as power differential; this study assumes that source field matches the paper definition. It is not reconstructed from future/outcome scores.

## Differences and interpretation

- Requested models 3 and 5; 6 and 7; and 9 and 10 share the same LR classifier-feature mapping based on the paper. Their equality is expected, not evidence for two independent model families.
- Standalone length-feature Logistic Regression is a new baseline, **not** a paper-experiment result.
- Partially printed Harbingers lexicon, tokenization, class-weight search, regularization, scaling and zero-missing handling are reproduction deviations.
- The authors' published scores are for their setup, not acceptance thresholds or results we should force. Large deviations should be treated as possible reproduction discrepancies, especially for Harbingers.
- The test split was scored after validation selection; published paper scores are displayed beside ours for context, not for tuning.

## Test and integrity checks

`PYTHONPATH=src pytest -q` completed successfully (25 passed, including all original Phase 1 tests and Phase 2 feature/baseline tests). `PYTHONPATH=src python -m diplomacy.data.pipeline --root .` completed successfully after artifact-path migration. SHA-256 digests of all three raw JSONL files were identical before/after Phase 1 rerun. The `data/` directory contains exactly `train.jsonl`, `validation.jsonl`, and `test.jsonl`.

## Future work (not Phase 3 implementation)

Retrieve the complete original Harbingers subjectivity lexicon and classifier configuration to enable a stricter reproduction. Investigate differences in validation selection and report stable per-game uncertainty before making scientific comparisons.
