from diplomacy.data.validator import validate_game_disjointness

def test_no_game_leakage():
    reports={'train':{'game_ids':[1,2]},'validation':{'game_ids':[3]},'test':{'game_ids':[4]}}
    assert validate_game_disjointness(reports)['passed']

def test_game_leakage_detected():
    reports={'train':{'game_ids':[1,2]},'validation':{'game_ids':[2]},'test':{'game_ids':[3]}}
    assert not validate_game_disjointness(reports)['passed']
