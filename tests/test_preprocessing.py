from diplomacy.data.preprocessing import write_jsonl,make_task_dataset
from diplomacy.data.parser import parse_conversation
import json

def test_task_output_round_trip(sample_conversation,tmp_path):
    records=make_task_dataset(parse_conversation(sample_conversation,'validation',0),'actual_lie')
    p=tmp_path/'processed.jsonl';write_jsonl(p,records)
    loaded=[json.loads(line) for line in p.read_text().splitlines()]
    assert loaded==records
    assert [r['message'] for r in loaded]==['Message one','Message two']
