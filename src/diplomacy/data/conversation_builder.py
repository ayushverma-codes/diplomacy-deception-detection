"""Build ordered message records while preserving conversation and game identities."""
from .parser import parse_conversation

def build_messages(conversations: list[dict], split: str) -> list[dict]:
    """Keep JSONL conversation order and original within-conversation order."""
    return [message for index, record in enumerate(conversations)
            for message in parse_conversation(record, split, index)]

def game_timelines(messages: list[dict]) -> dict[int,list[dict]]:
    """Return timelines ordered by original game-level absolute indices.

    This is a separate view: the canonical processed datasets retain source order.
    """
    by_game: dict[int,list[dict]]={}
    for message in messages:
        by_game.setdefault(message['game_id'],[]).append(message)
    for timeline in by_game.values():
        timeline.sort(key=lambda m:(int(m['absolute_message_index']),m['conversation_index'],m['message_position']))
    return by_game
