import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import pytest
@pytest.fixture
def sample_conversation():
    return {'game_id':15,'players':['italy','germany'],
        'messages':['Message one','Message two'],
        'sender_labels':[False,True],'receiver_labels':['NOANNOTATION',True],
        'speakers':['italy','germany'],'receivers':['germany','italy'],
        'absolute_message_index':[4,7],'relative_message_index':[0,1],
        'seasons':['Spring','Spring'],'years':['1901','1901'],
        'game_score':['3','3'],'game_score_delta':['0','0']}
