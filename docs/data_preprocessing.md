# Preprocessing decisions and leakage protections

1. Read original JSONL in order and validate record type plus parallel array lengths before extraction.
2. Expand each two-player conversation into aligned messages. Keep original within-conversation message order; the separate `game_timelines` view orders by the game's absolute message index when chronological analysis is required.
3. Preserve source `message` and create separate `normalized_message` with only Unicode NFC and CRLF/CR newline canonicalization. Do not lowercase, strip punctuation, remove stopwords, or collapse whitespace.
4. Preserve sender/receiver labels exactly. For actual lie detection select only rows with boolean `sender_label`; for suspected lie detection select only rows with boolean `receiver_label`. `NOANNOTATION` is excluded only from suspected-lie task.
5. Map `False` to `target=1` and `True` to `target=0`; the negative class is not an annotation placeholder.
6. Keep game-level splits intact. Split integrity is tested and checked before writes. No vocabulary, normalization statistics, or any other learned transforms are fitted on validation/test data.
7. Leave original `data/{train,validation,test}.jsonl` untouched. Write derived output exclusively to `data/processed/` and reports to `data/reports/`. SHA-256 checksums are checked before and after the pipeline.
8. Numeric fields are retained as supplied in processed JSONL; numeric casts occur only for descriptive EDA. Do not silently repair invalid values.

Pipeline: `PYTHONPATH=src python -m diplomacy.data.pipeline --root .`
