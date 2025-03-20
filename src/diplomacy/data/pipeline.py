"""End-to-end Phase 1 pipeline; invoke from the repository root."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from .loader import SPLITS,load_split
from .conversation_builder import build_messages
from .preprocessing import make_task_dataset,write_jsonl
from .validator import validate_conversations,validate_game_disjointness

def run(project_root:Path, *, strict:bool=True)->dict:
    data=project_root/'data'
    paths={s:data/f'{s}.jsonl' for s in SPLITS}
    before={s:hashlib.sha256(p.read_bytes()).hexdigest() for s,p in paths.items()}
    records={s:load_split(data,s) for s in SPLITS}
    reports={s:validate_conversations(records[s],s) for s in SPLITS}
    leakage=validate_game_disjointness(reports)
    if strict and not leakage['passed']:raise ValueError(f'Game split leakage: {leakage}')
    if strict and any(r['array_alignment_problems'] or r['unexpected_sender_labels'] or r['unexpected_receiver_labels'] for r in reports.values()):
        raise ValueError('Critical data validation failures; inspect report before proceeding')
    statistics={};tasks=('actual_lie','suspected_lie')
    for split in SPLITS:
        rows=build_messages(records[split],split)
        statistics[split]={
            'games':reports[split]['game_count'], 'conversations':reports[split]['conversation_count'],
            'messages':len(rows),'actual_lies':sum(r['sender_label'] is False for r in rows),
            'truthful_sender_messages':sum(r['sender_label'] is True for r in rows),
            'suspected_lies':sum(r['receiver_label'] is False for r in rows),
            'receiver_truthful':sum(r['receiver_label'] is True for r in rows),
            'receiver_missing':sum(r['receiver_label']=='NOANNOTATION' for r in rows),
        }
        for task in tasks:
            selected=make_task_dataset(rows,task)
            statistics[split][f'{task}_usable']=len(selected)
            statistics[split][f'{task}_positive']=sum(r['target'] for r in selected)
            write_jsonl(project_root/'artifacts'/'processed'/task/f'{split}.jsonl',selected)
    all_issues=[issue for s in SPLITS for issue in reports[s]['issues']]
    summary={'splits':statistics,'total_games':len(set.union(*(set(reports[s]['game_ids']) for s in SPLITS))),
             'total_messages':sum(statistics[s]['messages'] for s in SPLITS)}
    report={'splits':reports,'game_split_integrity':leakage,'issue_totals':dict(Counter(x['code'] for x in all_issues)),
            'original_sha256_before':before}
    report['original_sha256_after']={s:hashlib.sha256(p.read_bytes()).hexdigest() for s,p in paths.items()}
    report['original_files_unchanged']=before==report['original_sha256_after']
    reports_dir=project_root/'artifacts'/'reports_phase1';reports_dir.mkdir(parents=True,exist_ok=True)
    (reports_dir/'dataset_statistics.json').write_text(json.dumps(summary,indent=2)+'\n')
    (reports_dir/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Dataset validation summary','','Original files are preserved in `data/*.jsonl`; no data is moved.','',
           '| Split | Games | Conversations | Messages | Actual lies | Suspected lies | Missing receiver annotations |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for s,v in statistics.items():
        lines.append(f"| {s} | {v['games']} | {v['conversations']} | {v['messages']} | {v['actual_lies']} | {v['suspected_lies']} | {v['receiver_missing']} |")
    lines += ['','## Integrity',f"- Game-disjoint splits: **{leakage['passed']}**",
              f"- SHA-256 unmodified raw files: **{report['original_files_unchanged']}**",
              '- Empty conversations are kept in the raw files (produce no message-level records).',
              '- Repeated text is reported as a possible duplicate, but not automatically removed.',
              '- Source JSONL order and within-conversation message order remain unchanged.','',
              '## Validation issue counts']
    for key,count in sorted(report['issue_totals'].items()):lines.append(f'- {key}: {count}')
    (reports_dir/'data_summary.md').write_text('\n'.join(lines)+'\n')
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('.'))
    args=parser.parse_args();print(json.dumps(run(args.root.resolve()),indent=2))
