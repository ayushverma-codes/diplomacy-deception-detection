# Original Diplomacy dataset documentation

Source: uploaded ACL 2020 paper, Peskov et al., *It Takes Two to Lie: One to Lie, and One to Listen*. No external dataset downloaded.

## Physical structure

The original files are `data/train.jsonl`, `data/validation.jsonl` and `data/test.jsonl`. **Each JSONL row is one two-player conversation**, not one game. The original game-level split must be preserved. The paper describes nine training, one validation and two test games.

## Actual observed record schema

All three splits have the same fields: `messages`, `sender_labels`, `receiver_labels`, `speakers`, `receivers`, `absolute_message_index`, `relative_message_index`, `seasons`, `years`, `game_score`, `game_score_delta`, `players`, `game_id`.

Except `players` (two players) and `game_id` (integer game identifier), each field is an aligned array with one element per message. `messages` are the original strings; `speakers`/`receivers` identify the countries in the specific two-player conversation. `absolute_message_index` gives the original within-game ordinal; `relative_message_index` gives the sequence position within the conversation. `seasons` and `years` indicate the move season and year. `game_score` and `game_score_delta` are numeric **strings** in the source; the paper describes power differential as sender score minus receiver score. We preserve these strings without coercion in processed output; analysis explicitly converts them to numeric.

**Critical:** source `sender_labels` and `receiver_labels` use Python/JSON booleans: `True` means truthful and `False` means lie/suspected lie. `receiver_labels` additionally includes the literal `NOANNOTATION` for missing receiver annotations. The task-specific processed `target` is **1 for lie** and **0 for truth**, and the original boolean fields remain available.

## Field mapping

`messages[i] -> message`, `speakers[i] -> speaker`, `receivers[i] -> receiver`, `sender_labels[i] -> sender_label`, `receiver_labels[i] -> receiver_label`, `game_score_delta[i] -> score_delta`, `seasons[i] -> season`, `years[i] -> year`; similarly named fields retain their original meanings. Generated `message_id = game_id:conversation_index:message_position` is an identifier within this exported dataset, not a source-provided ID. `normalized_message` is an additional lossless-for-most-text normalization (Unicode NFC and normalized line endings); the original message is always retained. `split`, `conversation_index`, `message_position`, and `target_name` are derived metadata. The original values are never rewritten.

## Caveats

Games are distinct within the provided splits, but players may appear across games. Conversations containing zero messages exist; these are kept in the raw source but do not yield message rows. Repeated literal message text is not proof of duplicate annotation, so it is counted without deletion. Correlations discovered in the game environment should not be generalized to real-world truth detection.
