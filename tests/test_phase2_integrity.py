import json
from pathlib import Path
from diplomacy.data.parser import parse_conversation


def test_message_order_and_ids():
    record={
      'game_id':20,'players':['italy','germany'],
      'messages':['A','B'],'sender_labels':[True,False],
      'receiver_labels':[True,'NOANNOTATION'],
      'speakers':['italy','germany'],'receivers':['germany','italy'],
      'absolute_message_index':[30,32],'relative_message_index':[0,1],
      'seasons':['Spring','Spring'],'years':['1901','1901'],
      'game_score':['3','3'],'game_score_delta':['0','2']}
    rows=parse_conversation(record,'train',2)
    assert [x['message_id'] for x in rows]==['20:2:0','20:2:1']
    assert [x['absolute_message_index'] for x in rows]==[30,32]
