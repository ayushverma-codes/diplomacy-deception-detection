from diplomacy.data.preprocessing import make_task_dataset
from diplomacy.data.parser import parse_conversation

def test_target_direction_and_missing_receiver(sample_conversation):
    rows=parse_conversation(sample_conversation,'train',0)
    actual=make_task_dataset(rows,'actual_lie');suspected=make_task_dataset(rows,'suspected_lie')
    assert len(actual)==2 and len(suspected)==1
    assert [x['target'] for x in actual]==[1,0]
    assert [x['target'] for x in suspected]==[0]
    assert actual[0]['receiver_label']=='NOANNOTATION'
