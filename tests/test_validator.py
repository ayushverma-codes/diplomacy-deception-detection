from diplomacy.data.validator import validate_conversations

def test_validation_reports_missing_receiver(sample_conversation):
    r=validate_conversations([sample_conversation],'train')
    assert r['message_count']==2
    assert r['missing_receiver_annotations']==1
    assert not r['array_alignment_problems']

def test_array_misalignment_flagged(sample_conversation):
    sample_conversation['sender_labels'].pop()
    r=validate_conversations([sample_conversation],'train')
    assert r['array_alignment_problems']==1
