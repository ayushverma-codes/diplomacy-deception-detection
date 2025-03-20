import json
import pytest
from diplomacy.data.loader import load_jsonl,load_split,DatasetLoadError

def test_valid_jsonl(tmp_path):
    p=tmp_path/'example.jsonl';p.write_text('\n'.join([json.dumps({'x':1}),json.dumps({'x':2})])+'\n')
    assert [x['x'] for x in load_jsonl(p)]==[1,2]

def test_bad_jsonl_strict_and_lenient(tmp_path):
    p=tmp_path/'data.jsonl';p.write_text('not-json\n{"a":2}\n')
    with pytest.raises(DatasetLoadError):list(load_jsonl(p))
    assert list(load_jsonl(p,strict=False))==[{'a':2}]

def test_unknown_split(tmp_path):
    with pytest.raises(ValueError):load_split(tmp_path,'development')
