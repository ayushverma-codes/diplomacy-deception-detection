# Phase 4 — Context-Aware Neural Deception Detection

## Status

**Complete for Phase 4 training, final held-out evaluation and local CPU verification.** All six architecture/task experiments were trained using Google Colab GPU, evaluated using their validation-selected best checkpoints, transferred to Windows, and verified for checkpoint loading and inference parity. Existing Phases 1–3 artifacts were retained. Phase 5 has not been started.

- Paper: Peskov et al. (2020), *It Takes Two to Lie: One to Lie, and One to Listen* (https://aclanthology.org/2020.acl-main.353/).
- Authors' repository: https://github.com/DenisPeskoff/2020_acl_diplomacy.
- The paper and authors' README establish a message-only LSTM and a hierarchical context LSTM, whose context encoder is **forward** over previous messages. The original repository uses AllenNLP. The precise context-LSTM configuration and model implementation were **not fully retrievable in this environment**: therefore Phase 4's numerical hyperparameters and game chunking should not be represented as an exact paper reproduction.

## Architecture inspection and research mapping

| Experiment family | Word/message encoder | Context encoder | Input to prediction | Research status |
|---|---|---|---|---|
| BiLSTM | Phase 3 200d frozen GloVe, 100d bidirectional LSTM per direction, masked max pooling | None | Target message only | Same architecture as Phase 3; a separate independent-control run, not a new model family |
| BiLSTM with concatenated context | Same BiLSTM | None; word-level concatenation | Previous three game messages + target, separated by ` <CTX> ` | **Experimental extension**, not identified as a distinct model in the paper |
| Hierarchical/Context LSTM | Same message-level BiLSTM and max pooling | Forward unidirectional LSTM, hidden size 100 | Chronologically ordered message vectors | Paper-inspired; Colab-friendly **16-message chunk** adaptation |

All models use binary cross-entropy-equivalent **weighted two-class cross-entropy** (`nn.CrossEntropyLoss`), a linear two-logit classification head, dropout 0.5, Adam with learning rate 0.003, batch sizes 32 for individual/concatenated messages and 2 game chunks for hierarchical, gradient clipping norm 1.0, up to 15 epochs, patience 5, random seed 1994, and validation **Macro F1** checkpoint selection. These numerical settings intentionally begin with Phase 3's verified configuration; the *hierarchical* and *concatenated-context* settings are Phase 4 engineering choices, not verified original-paper hyperparameters.

Original Context LSTM: the ACL paper describes encoding each message's token sequence, then running a **forward LSTM across messages** so a message's classification uses earlier dialogue. The present hierarchical implementation uses bidirectional **within-message** word encoding and unidirectional **across-message** context. This prevents future-message leakage through the context encoder, but the 16-message chunk boundary resets hidden state and therefore **does not reproduce the paper's entire prior-message history**. To reproduce an unrestricted previous-message context, the chunking/training-memory strategy must be revised.

The concatenation experiment defaults to `before=3`, `after=0`, `include_target=True`, and a token limit of 256. Changing `after` to greater than zero yields a *noncausal* experiment that requires future messages at inference. Its predictions must not be described as real-time or purely preceding-context detection.

## Data preparation and integrity

- Raw original datasets: `data/{train,validation,test}.jsonl` — **never rewritten**.
- Existing Phase 1 outputs: `artifacts/processed/{actual_lie,suspected_lie}/{train,validation,test}.jsonl`.
- `chronological_games` reconstructs **all** game messages from the raw conversations, using original `absolute_message_index` and stable conversation/message tie-breaking.
- Processed `message_id` is aligned against raw IDs in `game_id:conversation_index:message_position` form. All processed target IDs must match a raw message.
- Never mix games or splits. The existing 9/1/2 game-level split is preserved.
- `target=1` means lie / suspected lie; `target=0` means truth / not suspected.
- For suspected-lie, `NOANNOTATION` is **not** converted to a negative target. Unannotated messages remain as *unlabeled context* with loss mask `-100`; they are excluded from supervised training and evaluation.
- Game IDs, message IDs and original indices are traceability metadata, not classifier features.
- The hierarchical collator creates tensors `[B, M, T]` for B game chunks, M messages and T dynamically padded tokens, plus word lengths and message masks. Padding is excluded from pooled message representations and from hierarchical context positions.
- Game context is ordered *across dialogues within the same game*; that choice resembles the paper's description of previous game messages but is not a guarantee of the authors' exact reader policy.

## GloVe strategy and Colab portability

**Default: `embeddings.mode=phase3_glove_reuse`.** The supplied, previously trained Phase 3 **actual-lie best checkpoint** already includes a frozen pretrained GloVe Twitter 27B 200d embedding matrix. `phase4_embeddings.initialize_from_phase3()` copies matching token vectors into each **training-only** Phase 4 vocabulary. Tokens absent from the Phase 3 training vocabulary receive deterministic random initial vectors. The original pretrained weights are frozen for Phase 4 too. This is a documented efficient reuse strategy: it avoids a repeat 1.5 GB download, but is **not mathematically identical to loading the full original GloVe archive for all tokens**. Check `embedding_info.copied_tokens` and `new_token_count` in the saved checkpoint.

For a closer direct-GloVe run, set `embeddings.mode` to `glove` in a Phase 4 YAML configuration and supply a verified archive at `artifacts/embeddings/glove/glove.twitter.27B.zip`; keep the existing checkpoints and experiment IDs separate before changing a configuration after training has started. Random embeddings are also supported via explicit mode `random` for smoke testing only; this is not paper reproduction.

The Colab notebook imports **the same production modules** that are used for local training, evaluation and inference. No notebook implementation of a neural model or training loop is present. Copy/save the project into Google Drive to persist checkpoints across disconnects. Free-tier GPU availability is not guaranteed and CPU fallback is supported.

## Files added

- `src/diplomacy/data/phase4_context.py` — chronological games, concatenation, game chunks and collator.
- `src/diplomacy/embeddings/phase4_embeddings.py` — explicit frozen GloVe reuse from Phase 3.
- `src/diplomacy/models/neural/phase4.py` — three architecture classes.
- `src/diplomacy/training/phase4_runner.py` — shared setup, training, masked evaluation, checkpoints, resume, portable inference, parity and comparisons.
- `configs/experiments/neural/phase_4/{bilstm,bilstm_concat_context,hierarchical_lstm}/{actual_lie,suspected_lie}.yaml` — six shared configurations.
- `tests/test_phase4_context.py` — synthetic CPU tests.
- `notebooks/12_phase_4_neural_models_colab.ipynb` — the **only** new notebook.
- `docs/phase_4_neural_models.md` — this document.

## Colab execution

1. Open `notebooks/12_phase_4_neural_models_colab.ipynb` in Colab.
2. Configure a GPU where available and upload the Phase 4 project ZIP to `/content/diplomacy-deception-detection-phase4.zip` or use an existing project folder in Google Drive. The notebook mounts Drive and installs only missing Python dependencies.
3. The notebook uses `ROOT = /content/drive/MyDrive/diplomacy_lstm/diplomacy-deception-detection` unless changed in the first setup cell. The project already includes the Phase 3 pretrained checkpoints needed for default GloVe reuse.
4. Choose `SELECTED_ARCHITECTURES`, `SELECTED_TASKS` (either all six sequentially, as executed here, or one at a time for Free Tier reliability), and `ACTION='train'`; keep `RESUME=True` after an interruption. The completed run used the existing trained checkpoints when available. CUDA RNG restore was corrected to convert saved RNG states to CPU `torch.ByteTensor` before calling `torch.cuda.set_rng_state_all()`.
5. Compare **validation** Macro F1 to select models/checkpoints. Set `RUN_FINAL_TEST=True` only after selection. Save parity references and download or sync all generated Phase 4 checkpoints.
6. The notebook reads Phase 2/3 metrics from `artifacts/` and never overwrites their results.

The completed Colab execution ran all six selected architecture/task pairs sequentially with `RESUME=True`, then switched to `ACTION="evaluate"` and `RUN_FINAL_TEST=True`. For a new run, Colab Free Tier users may split training across sessions if resource limits intervene. GloVe vector reuse avoids downloading large embedding archives again.

## Local run commands

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
python -m pytest tests/ -v
python -m diplomacy.training.phase4_runner --root . --architecture bilstm --task actual_lie --action train
python -m diplomacy.training.phase4_runner --root . --architecture bilstm_concat_context --task actual_lie --action train
python -m diplomacy.training.phase4_runner --root . --architecture hierarchical_lstm --task actual_lie --action train
```

The same commands support `--task suspected_lie`, `--action evaluate --split validation`, and `--action evaluate --split test`. Training on CPU is supported, but Google Colab GPU is recommended for full experiments.

### Inference after copying checkpoints from Drive

```powershell
python -m diplomacy.training.phase4_runner --root . --architecture bilstm --task actual_lie --action predict --text "I will support you."
python -m diplomacy.training.phase4_runner --root . --architecture bilstm_concat_context --task suspected_lie --action predict --context "We should cooperate." --text "I will support you."
python -m diplomacy.training.phase4_runner --root . --architecture hierarchical_lstm --task actual_lie --action predict --context "We should cooperate." --text "I will support you."
python -m diplomacy.training.phase4_runner --root . --architecture hierarchical_lstm --task actual_lie --action parity
```

For hierarchical local inference, provide a **chronologically ordered sequence** of messages from one game; boundaries follow the configured `max_messages` chunks. The CLI example uses two messages for demonstration. `predict_conversation(root, task, messages)` returns one prediction per message. The concatenated-context predictor must be given its preceding messages, with the same window policy as training.

## Artifacts and recovery

Each model/task gets a separate subdirectory under `artifacts/models/neural/phase_4/<architecture>/<task>/`. It contains `best_model.pt` (validation-selected weights), `latest_checkpoint.pt` (resume state), `history.json`, `training_summary.json`, and optional `parity_reference.json`.

The atomic latest checkpoint contains model/optimizer state, seed and RNG state, configuration, vocabulary, embedding information, epoch and history. No previously selected best checkpoint is overwritten unless validation Macro F1 improves. Evaluation saves JSON metrics and per-message CSV predictions under `artifacts/metrics/phase_4/` and `artifacts/predictions/phase_4/`. It records accuracy, precision, recall, minority F1, Macro F1, PR-AUC, weighted validation/test loss and confusion matrix. Metadata includes original game/message indices for hierarchical predictions.

**Resume caution:** A resumed experiment must use the same configuration and vocabulary. Do not modify the six configuration files midway through training. For a genuinely new configuration, use a new experiment directory or archive previous Phase 4 runs first.

## Experiment results

The values below are transcribed from the **executed Phase 4 Colab notebook** (`12_phase_4_neural_models_colab_all_six (1).ipynb`), including its saved model-comparison output, rather than being estimates. Validation checkpoint selection used Macro F1; final held-out test evaluation was performed after selecting the six best checkpoints. The comparison notebook reads Phase 2 and Phase 3 results from previously saved `artifacts/` files.

### Validation results (best checkpoints)

| Architecture | Actual Macro F1 | Actual positive-class F1 | Suspected Macro F1 | Suspected positive-class F1 |
|---|---:|---:|---:|---:|
| Message-only BiLSTM | 0.5144 | 0.0899 | 0.5490 | 0.1304 |
| Concatenated-context BiLSTM | 0.5471 | 0.1250 | **0.5666** | **0.1724** |
| Hierarchical Context LSTM | **0.5478** | **0.1374** | 0.5245 | 0.0702 |

The strongest validation model was hierarchical for actual-lie Macro F1 and concatenated context for suspected-lie Macro F1. This ranking **did not persist on the test sets**; conclusions must use held-out results, not validation alone.

### Final Phase 4 test results

| Architecture | Task | Accuracy | Macro F1 | Positive-class F1 |
|---|---|---:|---:|---:|
| Message-only BiLSTM | Actual lie | 0.871215 | **0.547854** | **0.165485** |
| Message-only BiLSTM | Suspected lie | 0.910303 | **0.517745** | 0.082645 |
| Concatenated-context BiLSTM | Actual lie | 0.874134 | 0.527172 | 0.122137 |
| Concatenated-context BiLSTM | Suspected lie | 0.884040 | 0.513484 | **0.088889** |
| Hierarchical Context LSTM | Actual lie | 0.885443 | 0.533237 | 0.127778 |
| Hierarchical Context LSTM | Suspected lie | 0.923636 | 0.485378 | 0.010471 |

The full per-experiment saved metrics, prediction probabilities and confusion matrices live in `artifacts/metrics/phase_4/`, `artifacts/predictions/phase_4/`, and `artifacts/figures/phase_4/`. Precision/recall/PR-AUC and losses are stored there; this summary intentionally does not invent unobserved values.

### Previous-phase comparison: Actual lie detection (held-out test)

| Phase | Model | Accuracy | Macro F1 | Lie F1 |
|---|---|---:|---:|---:|
| 2 | BoW + Power (model 09; model 10 ties) | 0.771251 | 0.547246 | **0.228782** |
| 3 | Standard LSTM | 0.854433 | 0.531203 | 0.141935 |
| 4 | Message-only BiLSTM | 0.871215 | **0.547854** | 0.165485 |
| 4 | Concatenated-context BiLSTM | 0.874134 | 0.527172 | 0.122137 |
| 4 | Hierarchical Context LSTM | 0.885443 | 0.533237 | 0.127778 |

### Previous-phase comparison: Suspected lie detection (held-out test)

| Phase | Model | Accuracy | Macro F1 | Suspected-lie F1 |
|---|---|---:|---:|---:|
| 2 | BoW + Power (model 09; model 10 ties) | 0.912323 | 0.511333 | 0.068670 |
| 3 | Standard LSTM | 0.880808 | 0.517024 | **0.097859** |
| 4 | Message-only BiLSTM | 0.910303 | **0.517745** | 0.082645 |
| 4 | Concatenated-context BiLSTM | 0.884040 | 0.513484 | 0.088889 |
| 4 | Hierarchical Context LSTM | 0.923636 | 0.485378 | 0.010471 |

### Findings and error analysis

1. **Message-only BiLSTM ranked highest in Phase 4 by test Macro F1** for both actual-lie and suspected-lie tasks. The actual-lie improvement over the Phase 2 BoW + Power Macro F1 is small (0.000608 absolute), and the suspected-lie improvement over Phase 3 is similarly small (0.000721); do not interpret these as proven generalizable improvements without uncertainty estimates.
2. **Adding context did not improve held-out Macro F1** in the configurations tested, despite concatenated context obtaining the strongest suspected-lie validation Macro F1. Possible explanations, not proven causes, include overfitting, small numbers of independent games, noisy context, truncation, and mismatched context length. Further controlled ablations would be necessary.
3. **Accuracy is misleading in imbalanced deception detection.** The hierarchical suspected-lie model achieved 92.36% accuracy while its positive-class F1 was only 1.05%; it largely fails to identify minority suspected lies at the chosen decision threshold.
4. **The minority class tells a different story.** The Phase 2 BoW + Power model retained the strongest actual-lie F1 (0.228782), while the Phase 3 Standard LSTM retained the strongest suspected-lie F1 (0.097859) among the comparison rows.
5. The executed notebook saved up to 50 incorrectly classified test rows per experiment in `artifacts/reports/phase_4/`. Error counts printed in the notebook were 353, 222, 345, 287, 314 and 189 respectively for the models in the test-results table order. These are sample-level misclassification counts, not explanations of why any specific decision failed. The analysis should not be framed as a causal explanation of deceptive language.
6. A complete experiment ledger should preserve the individual validation curves, test confusion matrices and per-message probabilities, since the ranking of models changes depending on metric and task.

## Testing and limitations

The combined Phase 1–4 CPU test suite passed **45/45 tests** on local Windows Python 3.14.7 (reported runtime 8.14 seconds). After transferring the six trained Colab checkpoints to Windows, all six `best_model.pt` files were found, all **six** parity commands passed using three reference predictions each (**18/18**), and local CPU prediction commands succeeded for message-only, concatenated-context and hierarchical models. The CPU unit suite covers game isolation, chronological order, target alignment, missing annotations, variable-length collation, masking, padding invariance, causal hierarchical directionality, one training step, weighted metrics, checkpoint loading, CPU inference, and saved-reference parity. A separate **six-experiment synthetic smoke run** trained, evaluated and verified parity for every architecture/task combination. This was functionality testing; full-data GPU training and held-out test evaluation were subsequently completed independently in Colab.

Limitations to report in any paper or interview:

1. **Message-only Phase 4 BiLSTM duplicates the Phase 3 architecture**, though independent runs/checkpoints are supported.
2. **Concatenated text is an added experimental model**, not confirmed in the ACL paper.
3. **Hierarchical game chunking resets context between chunks**, limiting historical context to at most the current chunk; the original model's exact context reader and dimensions were not fully verified.
4. Pretrained embedding reuse is only exact for token overlap with the existing Phase 3 checkpoint vocabulary.
5. One task's unannotated messages are kept as context but never treated as negative labels.
6. Only training data builds the vocabulary; game splits remain fixed. In-context histories within the same game are allowed, but cross-game context or post-label information is not.
7. Google Colab Free Tier hardware and runtime availability remain variable. GPU training and the displayed evaluation results are grounded in the executed Colab notebook for this run; CPU parity was then tested separately on Windows.

### Completed local CPU verification

After copying the Phase 4 outputs into an existing Phase 1–3 Windows project:

```powershell
$env:PYTHONPATH = 'src'
python -m pytest tests/ -v

foreach ($arch in 'bilstm','bilstm_concat_context','hierarchical_lstm') {
    foreach ($task in 'actual_lie','suspected_lie') {
        python -m diplomacy.training.phase4_runner --root . --architecture $arch --task $task --action parity --device cpu
        if ($LASTEXITCODE -ne 0) { throw "Parity failed: $arch / $task" }
    }
}
```

Observed: `45 passed in 8.14s`, followed by `{'checked': 3, 'passed': True}` for each of six combinations. Individual sample CPU predictions for the sentence *“I promise I will support your attack next season.”* were 0.518114 for message-only BiLSTM, 0.364515 for concatenated context and 0.484568 for the **target** message in the hierarchical model; context-based models were supplied with the preceding question *“Will you support my attack next season?”*. These are illustrative classifier probabilities, **not proof of deception**. Hierarchical prediction additionally returns an output for the preceding message.

### Completion status

Phase 4 is finished for implementation, all six full-data training/evaluation runs, Colab artifacts, local unit tests and local inference/parity verification. The completed experiments do not establish exact architectural reproduction of the original paper, nor statistical superiority of one model. Phase 5 is **not** implemented and has not been started.
