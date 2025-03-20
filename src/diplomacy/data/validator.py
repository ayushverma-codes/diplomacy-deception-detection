"""Non-mutating, audit-oriented checks of conversation records and messages."""
from collections import Counter
from .parser import REQUIRED_FIELDS, MESSAGE_FIELDS


def validate_conversations(records: list[dict], split: str) -> dict:
    """Return detailed issues and statistics, retaining empty conversations."""
    issues=[]; seen_representations=set(); seen_ids=set(); count=0
    unique_players=set(); empty_messages=0; missing_receiver=0; duplicate_text=0
    seen_message_text=set(); unexpected_sender=Counter(); unexpected_receiver=Counter()
    games=set(); per_game=Counter(); per_speaker=Counter(); per_receiver=Counter()
    duplicate_indices=0; index_decreases=0; alignment_problems=0
    for rid,record in enumerate(records):
        prefix={'split':split,'conversation_index':rid,'game_id':record.get('game_id')}
        missing=[key for key in REQUIRED_FIELDS if key not in record]
        if missing:
            issues.append({**prefix,'code':'missing_fields','details':missing});continue
        if not isinstance(record['game_id'],int) or isinstance(record['game_id'],bool):
            issues.append({**prefix,'code':'invalid_game_id','details':str(record['game_id'])})
        else:games.add(record['game_id'])
        lengths={k:len(record[k]) if isinstance(record[k],list) else -1 for k in MESSAGE_FIELDS}
        if len(set(lengths.values()))!=1 or -1 in lengths.values():
            alignment_problems+=1
            issues.append({**prefix,'code':'array_alignment','details':lengths});continue
        if not isinstance(record['players'],list) or len(record['players'])!=2:
            issues.append({**prefix,'code':'invalid_players','details':str(record['players'])})
        else: unique_players.update((record['game_id'],p) for p in record['players'])
        n=lengths['messages'];count+=n;per_game[record['game_id']]+=n
        if n==0:issues.append({**prefix,'code':'empty_conversation','details':'0 messages'})
        # Repeated conversation arrays may be valid only if distinct; flag identical records.
        sig=(record['game_id'],tuple(str(v) for v in record['absolute_message_index']))
        if n and sig in seen_representations:issues.append({**prefix,'code':'duplicate_conversation_indices','details':'same index sequence'})
        if n:seen_representations.add(sig)
        prev_abs=None; prev_rel=None
        for i in range(n):
            msg=record['messages'][i]
            if not isinstance(msg,str) or not msg.strip():
                empty_messages+=1;issues.append({**prefix,'message_position':i,'code':'empty_message','details':repr(msg)})
            elif (record['game_id'],msg) in seen_message_text:duplicate_text+=1
            seen_message_text.add((record['game_id'],msg)) if isinstance(msg,str) else None
            sl=record['sender_labels'][i];rl=record['receiver_labels'][i]
            if type(sl) is not bool:unexpected_sender[str(sl)]+=1;issues.append({**prefix,'message_position':i,'code':'invalid_sender_label','details':repr(sl)})
            if rl=='NOANNOTATION':missing_receiver+=1
            elif type(rl) is not bool:unexpected_receiver[str(rl)]+=1;issues.append({**prefix,'message_position':i,'code':'invalid_receiver_label','details':repr(rl)})
            sender=record['speakers'][i];receiver=record['receivers'][i]
            per_speaker[str(sender)]+=1;per_receiver[str(receiver)]+=1
            if sender==receiver:issues.append({**prefix,'message_position':i,'code':'self_message','details':str(sender)})
            if isinstance(record['players'],list) and (sender not in record['players'] or receiver not in record['players']):
                issues.append({**prefix,'message_position':i,'code':'player_alignment','details':f'{sender} -> {receiver}'})
            ai=record['absolute_message_index'][i];ri=record['relative_message_index'][i]
            try:
                ai=int(ai);ri=int(ri)
                key=(record['game_id'],ai)
                if key in seen_ids:
                    duplicate_indices+=1;issues.append({**prefix,'message_position':i,'code':'duplicate_absolute_index','details':ai})
                seen_ids.add(key)
                if prev_abs is not None and ai<=prev_abs:
                    index_decreases+=1;issues.append({**prefix,'message_position':i,'code':'non_increasing_absolute_index','details':ai})
                if prev_rel is not None and ri<=prev_rel:
                    issues.append({**prefix,'message_position':i,'code':'non_increasing_relative_index','details':ri})
                prev_abs=ai;prev_rel=ri
            except (ValueError,TypeError):
                issues.append({**prefix,'message_position':i,'code':'invalid_index','details':[repr(ai),repr(ri)]})
            for num_field in ('years','game_score','game_score_delta'):
                try:int(record[num_field][i])
                except (ValueError,TypeError):issues.append({**prefix,'message_position':i,'code':'invalid_numeric_field','details':num_field})
    return {'split':split,'conversation_count':len(records),'game_ids':sorted(games),
            'game_count':len(games),'message_count':count,'empty_message_count':empty_messages,
            'missing_receiver_annotations':missing_receiver,'duplicate_message_text_count':duplicate_text,
            'duplicate_absolute_index_count':duplicate_indices,'non_increasing_absolute_index_count':index_decreases,
            'array_alignment_problems':alignment_problems,'unexpected_sender_labels':dict(unexpected_sender),
            'unexpected_receiver_labels':dict(unexpected_receiver),'unique_player_game_pairs':len(unique_players),
            'unique_speakers':sorted(per_speaker),'unique_receivers':sorted(per_receiver),
            'messages_per_game':dict(per_game),'messages_per_speaker':dict(per_speaker),
            'messages_per_receiver':dict(per_receiver),'issues':issues}

def validate_game_disjointness(split_reports: dict) -> dict:
    names=list(split_reports); intersections={}
    for i,a in enumerate(names):
        for b in names[i+1:]:
            intersections[f'{a}__{b}']=sorted(set(split_reports[a]['game_ids']) & set(split_reports[b]['game_ids']))
    return {'passed':not any(intersections.values()),'intersections':intersections}
