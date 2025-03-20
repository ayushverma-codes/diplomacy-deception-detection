import pytest
from diplomacy.data.parser import parse_conversation, RecordAlignmentError

def test_parser_alignment(sample_conversation):
    rows=parse_conversation(sample_conversation,'train',3)
    assert len(rows)==2
    assert rows[0]['message_id']=='15:3:0'
    assert (rows[1]['speaker'],rows[1]['receiver'])==('germany','italy')
    assert rows[0]['score_delta']=='0'
    assert rows[0]['sender_label'] is False

def test_misalignment_fails(sample_conversation):
    sample_conversation['receivers'].pop()
    with pytest.raises(RecordAlignmentError):parse_conversation(sample_conversation,'train',0)
