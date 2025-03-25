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

## Validation and results status

- CPU unit tests and a synthetic 1-epoch training/checkpoint/inference/resume test are included. Check actual terminal test output for counts.
- **Full Colab GPU training: not run here.** Actual-lie and suspected-lie final validation/test results, plots and timing are pending execution by the user in Colab. Never present unrun metrics as achieved.

## Limitations

Tokenizer and token truncation differ from original AllenNLP; exact original suspected-lie configuration needs verification. Random vector fallback is a deviation. The corpus is small and severely imbalanced, and has clustered game-level observations. All experiment results should include their true embedding mode. This phase never reads conversational context (reserved for Phase 4).
