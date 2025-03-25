# Phase 3 — Standard Message-Level LSTM

## Method and original sources

Original paper: Peskov et al., ACL 2020, *It Takes Two to Lie: One to Lie, and One to Listen*.
Official repository: https://github.com/DenisPeskoff/2020_acl_diplomacy
Original actual-lie configuration: https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/master/configs/actual_lie/lstm.jsonnet

The verified actual-lie `lstm.jsonnet` uses lowercased single-ID tokens, frozen GloVe Twitter 27B 200-dimensional embeddings, a single bidirectional LSTM (hidden size 100/direction), max pooling, dropout 0.5, an output classifier, Adam learning rate 0.003, batch size 32, 15 epochs, patience 5, gradient clipping 1.0, seed 1994, and weighted loss positive weight 30. It selects on validation macro F1.

The original repository used AllenNLP and `utils/singlemessage_format.py`. We implement a plain PyTorch replacement using the existing Phase 1 processed message-level JSONL files. This does **not** include conversation history, metadata or power features in predictions. Each message has an independent forward pass.

**Verified actual-lie settings vs implementation decisions:** The embedding type, dimensions, frozen trainability, LSTM bidirectionality, max aggregation, loss weighting, learning rate, batch size, patience and training seed follow the published actual-lie config. Our regular-expression word/punctuation tokenizer is a documented approximation of the original AllenNLP tokenizer (not exact tokenization parity). The original repo's exact suspected-lie LSTM config was not confirmed; the suspected-lie YAML initially carries over the verified actual-lie settings as an explicit adaptation. `max_tokens=256` and `max_vocab_size=30000` are **Colab resource safeguards**, not established original settings; truncated texts can remove useful signals. Vocabulary is built on training data only. Pretrained OOV word vectors use seeded normal random initialization, and PAD is zero. The full GloVe Twitter archive is ~1.4 GB and may require significant Colab storage/network bandwidth.

A user can select `embeddings.mode=random` in a shared task YAML for a short CPU/GPU smoke test, but this **changes the experiment from the paper's pretrained setting**. The default YAML deliberately requests `glove`. GloVe acquisition is opt-in (`--download-glove` or notebook variable), to avoid surprise large downloads. A successful full experiment requires the GloVe resource or an explicitly documented changed YAML.

## Dataset and training

- Original `data/` contains exactly train.jsonl, validation.jsonl, and test.jsonl.
- Processed message datasets are under `artifacts/processed/{actual_lie,suspected_lie}/`.
- Targets: 0 truthful/not suspected, 1 actual or suspected lie. Missing receiver annotations are absent from suspected-lie datasets but preserved for actual-lie.
- Training and validation are game-disjoint. No label or message history is supplied to model forward.
- Training uses weighted cross-entropy (30 for positive class by default), gradient clipping, epoch-level validation, patience-based early stopping, latest checkpoint and best checkpoint. Model selection uses validation Macro F1; the test set is evaluated only after training.
- Output state_dict checkpoints include vocabulary, model config, training settings, embedding source information, class map, software versions, seed and RNG states. Resume reads the latest checkpoint and restores optimizer state and RNG states.
- Results include accuracy, precision, recall, lie F1, macro F1, PR-AUC, confusion matrix and loss. Predictions are saved with message and game IDs, labels and probabilities. Models are independent per task.

## Google Colab

Open `notebooks/11_standard_lstm_colab.ipynb`, choose GPU in Runtime settings if available, mount Google Drive, upload the Phase 3 project ZIP (or set a GitHub URL) and run sequentially. The notebook extracts or clones the project into a directory on Google Drive, installs missing dependencies and calls only the project modules for model logic. Best and latest checkpoints are saved directly to Drive. Runtime restarts can resume a specific task from its latest checkpoint. The GPU is optional; CPU works but is slower.

Avoid extracting more than one nested project directory. On reconnect, keep the same project folder and use `resume=True`. Validate the checkpoint and vocabulary before evaluating on test. All generated outputs stay under `artifacts/`.

## Local CPU inference / prediction parity

From project root (Windows PowerShell):

```powershell
$env:PYTHONPATH = 'src'
python -m diplomacy.training.standard_lstm_runner --root . --task actual_lie --action predict --text 'I promise I will help you.'
python -m diplomacy.training.standard_lstm_runner --root . --task actual_lie --action evaluate --split test
```

Or use `from diplomacy.inference.standard_lstm import load_predictor`, then `load_predictor(path,device='cpu').predict(message)`. Transfer the **complete** Google Drive project artifact folder to the local repository if checkpoints do not already exist. `save_parity_reference` creates a JSON reference of a small number of Colab predictions; after moving the checkpoint, `verify_parity_reference` checks CPU outputs against the reference to absolute tolerance 1e-4. Exact numeric parity across GPU/CPU is not guaranteed.

## Phase 2 comparison

`diplomacy.training.standard_lstm_runner.compare_phase2(root, task)` loads Phase 2 test metrics directly from artifacts and appends the actual Phase 3 test metrics when available. Do not infer missing results. Accuracy is insufficient given class imbalance; inspect Macro F1 and positive-class F1. Look for overfitting in the training history and evaluate confusion matrices and false positives/negatives.

## Executed experiments and results (November 2026)

The user executed `notebooks/11_standard_lstm_colab.ipynb` on Google Colab with a **Tesla T4 GPU** (PyTorch reported `2.14.1+cu130`). Both experiments used the `glove` embedding mode with the ZIP member `glove.twitter.27B.200d.txt`, 200-dimensional frozen embeddings, bidirectional LSTM (100 hidden units per direction), max pooling, and the shared task YAML configurations. These numbers are drawn from the executed notebook, rather than inferred from the paper.

### Dataset sizes and pretrained embedding coverage

| Task | Train | Validation | Test | Training vocabulary | GloVe covered tokens | Recorded coverage |
|---|---:|---:|---:|---:|---:|---:|
| Actual lie | 13,132 | 1,416 | 2,741 | 7,902 | 7,293 | 92.32% |
| Suspected lie | 12,025 | 1,283 | 2,475 | 7,654 | 7,083 | 92.56% |

These are the counts/coverage reported by the executed notebook. Receiver messages with missing labels are excluded from the suspected-lie dataset only.

### Validation results and model selection

Both experiments used **validation Macro F1** to select `best_model.pt`. Both selected **epoch 8** and subsequently stopped after epoch 13 with the configured patience. The reported loss is the weighted validation loss returned by the evaluator.

| Metric | Actual lie | Suspected lie |
|---|---:|---:|
| Accuracy | 0.889831 | 0.925955 |
| Precision (positive class) | 0.061404 | 0.112903 |
| Recall (positive class) | 0.125000 | 0.148936 |
| Positive-class F1 | 0.082353 | 0.128440 |
| Macro F1 | 0.511875 | 0.544888 |
| PR-AUC | 0.053896 | 0.062916 |
| Weighted validation loss | 1.147077 | 1.188733 |
| Best epoch | 8 | 8 |

Validation confusion matrices are shown as `[[TN, FP], [FN, TP]]`, with class `1` meaning lie / suspected lie:

- Actual lie: `[[1253, 107], [49, 7]]`.
- Suspected lie: `[[1181, 55], [40, 7]]`.

### Final test results

The saved best-validation checkpoints were used for the test evaluation. Results are based on 2,741 actual-lie messages and 2,475 suspected-lie messages.

| Metric | Actual lie | Suspected lie |
|---|---:|---:|
| Accuracy | 0.854433 | 0.880808 |
| Precision (positive class) | 0.146667 | 0.098765 |
| Recall (positive class) | 0.137500 | 0.096970 |
| Positive-class F1 | 0.141935 | 0.097859 |
| Macro F1 | **0.531203** | **0.517024** |
| PR-AUC | 0.131012 | 0.084380 |
| Best-validation epoch | 8 | 8 |

Do **not** interpret high accuracy as strong deception detection: the positive class is rare, and positive-class recall is low on both tasks.

### Comparison with the completed Phase 2 baseline runs

The notebook loaded the saved Phase 2 test metrics rather than hardcoding them. The table below uses the already executed Phase 2 experiments.

| Model | Actual-lie Macro F1 | Actual-lie positive F1 | Suspected-lie Macro F1 | Suspected-lie positive F1 |
|---|---:|---:|---:|---:|
| Majority | 0.477108 | 0.000000 | 0.482759 | 0.000000 |
| BoW + Power | **0.547246** | **0.228782** | 0.511333 | 0.068670 |
| Standard LSTM | 0.531203 | 0.141935 | **0.517024** | **0.097859** |

- Actual-lie test Macro F1 is **0.016043 lower** than BoW + Power (about **1.60 percentage points**). Actual-lie positive-class F1 is also lower.
- Suspected-lie test Macro F1 is **0.005691 higher** than BoW + Power (about **0.57 percentage points**). Suspected-lie positive-class F1 is also higher.
- These are observed results of the saved configurations; they do not prove one architecture is universally superior. BoW + Power uses a game-state feature not available to the text-only Standard LSTM, so the comparison is not a controlled text-only architecture ablation.
- Validation metrics fluctuated after training loss decreased, suggesting possible overfitting / unstable minority-class generalization; additional controlled experiments would be needed to establish causes.

### Persisted output and qualitative analysis

The executed notebook confirmed the following files and directories for both tasks:

```text
artifacts/
├── models/neural/standard_lstm/
│   ├── actual_lie/{best_model.pt, latest_checkpoint.pt, parity_reference.json}
│   └── suspected_lie/{best_model.pt, latest_checkpoint.pt, parity_reference.json}
├── predictions/standard_lstm/
├── metrics/standard_lstm/
├── figures/standard_lstm/
└── reports/phase_3/
    ├── actual_lie_test_errors.csv
    └── suspected_lie_test_errors.csv
```

Test confusion matrix plots and training curves were generated by the executed notebook. The error-analysis CSVs include message-level misclassification records. Their examples should be treated as qualitative observations, not proof of particular linguistic causes of failure.

### Local checkpoint transfer and CPU verification (Windows)

After copying the Google Drive artifacts to `C:\Projects\diplomacy-deception-detection`, the user executed the inference commands from the project root in a Python **3.14.7** virtual environment with `$env:PYTHONPATH = "src"`. Both Colab-trained checkpoints loaded on CPU without retraining.

The test input was:

> I promise I will support your attack next season.

| Task | Predicted label | P(lie/suspected lie) | P(truth/not suspected) |
|---|---:|---:|---:|
| Actual lie | 0 | 0.1895957887 | 0.8104042113 |
| Suspected lie | 0 | 0.0945832729 | 0.9054167271 |

These probabilities reflect model outputs, not verified truth judgments.

Local reference-parity results from `check_phase3_parity.py`:

```text
actual_lie    checked=3  tolerance=0.0001  passed=True
suspected_lie checked=3  tolerance=0.0001  passed=True
```

Six reference predictions were checked (three per task), matching Colab within the stated absolute tolerance of **1e-4**. This verifies cross-environment inference compatibility for these samples; it is not a proof of bitwise GPU/CPU equality.

### Complete local regression test suite

The user executed the following commands from the project root in Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests/ -v
```

**Result: 35 tests passed in 14.09 seconds**, with no failures, using Python 3.14.7 and pytest 9.2.0. The suite includes existing Phase 1 and Phase 2 tests and Phase 3 tests for tokenization, vocabulary training-only isolation, embeddings, collator, padding invariance, target handling, weighted loss, metrics, checkpoint/resume, CPU inference, and parity-reference checks.

### Completion assessment

The implementation, Colab training for both tasks, checkpoint export and recovery, saved test evaluation, local CPU inference, prediction parity, and regression tests have been exercised successfully. Original raw-file integrity was checked in the Colab notebook. **Phase 3 is complete for the reported experiments**, subject to the scientific reproduction limitations below; no Phase 4 development is implied.

## Limitations

Tokenizer and token truncation differ from original AllenNLP; exact original suspected-lie configuration needs verification. Random vector fallback is a deviation. The corpus is small and severely imbalanced, and has clustered game-level observations. All experiment results should include their true embedding mode. This phase never reads conversational context (reserved for Phase 4).
