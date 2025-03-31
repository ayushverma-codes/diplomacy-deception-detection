# Phase 5 research design and source inspection

## Research provenance
- Paper: Denis Peskov et al., *It Takes Two to Lie: One to Lie, and One to Listen*, ACL 2020, https://aclanthology.org/2020.acl-main.353/
- Public authors' repository: https://github.com/DenisPeskoff/2020_acl_diplomacy
- Verified BERT original config: https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/master/configs/actual_lie/bert%2Bcontext.jsonnet
- Dataset schema: https://github.com/DenisPeskoff/2020_acl_diplomacy/blob/master/data/README.md

The paper describes a **forward context LSTM** over individual message representations; context is chronological and prediction uses previous messages. The original repo uses AllenNLP; this implementation is PyTorch. The paper finds Context LSTM + Power improves actual-lie F1, but BERT did not improve it. These are historical results, NOT Phase 5 test outcomes.

## Exact recovered BERT configuration
The published `configs/actual_lie/bert+context.jsonnet` names `bert-base-uncased`, lowercase tokenization, `max_pieces=20`, a `bert_pooler` message encoder with `requires_grad=true` (fine-tuned), 768-dimensional input to a forward conversation LSTM, context hidden size 200, dropout 0.4, batch size 4, Adam learning rate 0.0003, 15 epochs, patience 10, gradient clipping 1, original seed 1994, and positive-class weight 15. The message embedding is a BERT *pooler output*, not an arbitrary mean or last-layer CLS pooling. Our HuggingFace `AutoModel.pooler_output` approximates AllenNLP's historical `bert_pooler` behavior.

The other original `.jsonnet` configurations were not fully recovered in this inspection. Do not present their hyperparameters or this PyTorch architecture as an exact reproduction. The 8-message chunk length, batch size of 1 for BERT, explicit train-only score normalization, weight initialization from Phase 4, and power fusion after the context LSTM are documented experimental adaptations.

## Power mapping verified
The paper defines contemporaneous differential as *sender victory points minus receiver victory points*. The actual source files in this project have parallel arrays `game_score` (sender's score) and `game_score_delta` (sender-minus-recipient differential); the authors' data README also calls this concept `score_delta`. **The implementation reads `game_score_delta` from the original JSONL and never treats `game_score` as a substitute.** It validates array lengths and each score differential and retains message alignment. No final-game scores or label-derived features are used.

The scaler is fitted only on training examples; it imputes missing scores with the train median, then applies train-set mean/std normalization to validation, test, and inference inputs. The statistics are saved with the checkpoint. This preprocessing is an adaptation, not a verified feature of the original publication.

## Relationship to Phases 3 and 4
Phase 3 uses a 200-dimensional frozen GloVe message BiLSTM (100 hidden units per direction, max pooling). Phase 4's HierarchicalContextLSTM adds a forward 100-hidden context LSTM over chunks of up to 16 messages. Phase 5 reuses this working model class and its learned vocabulary/encoder as the no-power baseline; power is fused to the context hidden state immediately before classification. The BERT variants use the paper-supported 768-dimensional pretrained `bert-base-uncased` pooler and 200-hidden forward context LSTM. These branches intentionally have different message representations.

**Experiment semantics**: Phase 5 baseline is freshly trainable with weights initialized from Phase 4, NOT an independently random initialization. Comparisons must mention the transferred starting point. BERT and BERT+Power start from the same original pretrained BERT checkpoint; seeds and data/task splits are shared. All four are separately checkpointed by task.

## Data leakage controls and limitations
- Original game-level splits (9 train, 1 validation, 2 test games) are never altered.
- Sender labels are binary truth/lie; `NOANNOTATION` receiver labels are encoded -100 and excluded from supervised loss and evaluation but messages remain in the conversational encoder.
- The context LSTM is forward-only, but its state resets at 8-message chunks to fit Colab Free Tier. This limits historical context relative to full-game modelling.
- The BERT variants have a 20-token limit by default, matching the recovered original configuration; this can truncate long messages.
- This code uses frozen Phase 4 GloVe vectors for non-BERT variants and fine-tuned BERT for BERT variants. Their encoder capacities differ; compare cautiously.
- Validation Macro F1 selects checkpoints; test set is only used after all choices are fixed.
- Full Colab training, actual Phase 5 test metrics, and Colab/local parity are **not performed by this package**; they require executing the notebook.
