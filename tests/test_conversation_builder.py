from diplomacy.data.conversation_builder import build_messages,game_timelines

def test_original_order_preserved(sample_conversation):
    rows=build_messages([sample_conversation],'train')
    assert [r['absolute_message_index'] for r in rows]==[4,7]
    assert [r['message_position'] for r in rows]==[0,1]

def test_game_timeline_is_sorted_separately(sample_conversation):
    newer={**sample_conversation,'absolute_message_index':[20,23]}
    rows=build_messages([newer,sample_conversation],'train')
    assert [r['absolute_message_index'] for r in rows]==[20,23,4,7]
    assert [r['absolute_message_index'] for r in game_timelines(rows)[15]]==[4,7,20,23]
