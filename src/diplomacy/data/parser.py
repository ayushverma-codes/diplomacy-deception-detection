"""Convert a two-player conversation's aligned arrays into message records."""
from .normalizer import normalize_text
MESSAGE_FIELDS = ('messages','sender_labels','receiver_labels','speakers','receivers',
                  'absolute_message_index','relative_message_index','seasons','years',
                  'game_score','game_score_delta')
REQUIRED_FIELDS = (*MESSAGE_FIELDS,'game_id','players')

class RecordAlignmentError(ValueError):
    """One record lacks fields or has misaligned arrays."""

def parse_conversation(record: dict, split: str, conversation_index: int) -> list[dict]:
    """Preserve the original within-conversation order and original game timestamps.

    Original labels use False=lie and True=truth (never Python truthiness on NOANNOTATION).
    score_delta maps from the original game_score_delta field.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing: raise RecordAlignmentError(f'{split} conversation {conversation_index}: missing {missing}')
    if not all(isinstance(record[f], list) for f in MESSAGE_FIELDS):
        raise RecordAlignmentError(f'{split} conversation {conversation_index}: message fields must be lists')
    sizes = {field:len(record[field]) for field in MESSAGE_FIELDS}
    if len(set(sizes.values()))!=1:
        raise RecordAlignmentError(f'{split} conversation {conversation_index}: array lengths differ {sizes}')
    messages=[]
    for i in range(len(record['messages'])):
        source = record['messages'][i]
        messages.append({
            'split':split, 'game_id':record['game_id'], 'conversation_index':conversation_index,
            'message_id':f"{record['game_id']}:{conversation_index}:{i}",
            'message_position':i, 'message':source,
            'normalized_message':normalize_text(source) if isinstance(source,str) else None,
            'speaker':record['speakers'][i], 'receiver':record['receivers'][i],
            'sender_label':record['sender_labels'][i], 'receiver_label':record['receiver_labels'][i],
            'game_score':record['game_score'][i], 'score_delta':record['game_score_delta'][i],
            'season':record['seasons'][i], 'year':record['years'][i],
            'absolute_message_index':record['absolute_message_index'][i],
            'relative_message_index':record['relative_message_index'][i],
            'players':record['players'],
        })
    return messages
