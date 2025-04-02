"""CPU-only tests for Phase 6 analysis; no trained models required."""
import math
import importlib.util
from pathlib import Path
import pandas as pd
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'src/diplomacy/analysis/phase6_analysis.py'
spec = importlib.util.spec_from_file_location('phase6_analysis', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_confusion_matrix_recomputation():
    r=m.metric_from_cm([[90,10],[8,2]])
    assert r['precision']==pytest.approx(2/12)
    assert r['recall']==pytest.approx(2/10)
    assert r['lie_f1']==pytest.approx(4/22)
    assert r['accuracy']==pytest.approx(92/110)

def test_all_negative_f1_zero():
    r=m.metric_from_cm([[100,0],[10,0]])
    assert r['lie_f1']==0
    assert r['recall']==0
    assert r['accuracy']==pytest.approx(100/110)

def test_duplicate_predictions_rejected(tmp_path):
    p=tmp_path/'p.csv'
    p.write_text('message_id,game_id,target,prediction\nx,4,0,0\nx,4,1,0\n')
    with pytest.raises(ValueError,match='Duplicate'):
        m.load_pred(p)

def test_missing_prediction_fields_rejected(tmp_path):
    p=tmp_path/'p.csv';p.write_text('message_id,game_id,target\nx,4,0\n')
    with pytest.raises(ValueError,match='Missing prediction fields'):
        m.load_pred(p)

def test_jsonl_scores_loaded(tmp_path):
    p=tmp_path/'p.jsonl';p.write_text('{"message_id":"x","game_id":4,"target":1,"prediction":0,"score":0.2}\n')
    d=m.load_pred(p)
    assert len(d)==1 and d.iloc[0]['prob_lie']==pytest.approx(0.2)
