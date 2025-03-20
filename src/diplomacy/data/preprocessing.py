"""Prepare two independently annotated task datasets without label leakage."""
import json
from pathlib import Path

VALID_BOOLEAN_LABELS = (True,False)

def make_task_dataset(messages: list[dict], task: str) -> list[dict]:
    """Select valid annotations for the chosen task; retain both original labels.

    True signifies truthful, False signifies deceptive; target 1 signifies deception.
    """
    target_field={'actual_lie':'sender_label','suspected_lie':'receiver_label'}.get(task)
    if target_field is None:raise ValueError(f'Unknown task {task!r}')
    return [{**row,'target':int(row[target_field] is False),'target_name':task}
            for row in messages if type(row[target_field]) is bool]

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
