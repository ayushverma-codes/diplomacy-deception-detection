"""Shared Phase 3 orchestration. The Colab notebook calls these functions."""
import argparse
import csv
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from diplomacy.data.text_dataset import Vocabulary, MessageDataset, load_rows
from diplomacy.data.text_collator import collate_messages
from diplomacy.embeddings.glove_embeddings import initialize_embeddings
from diplomacy.models.neural.lstm import StandardLSTM
from diplomacy.training.lstm_trainer import (seed_everything, train, evaluate,
                                             build_loss, load_checkpoint)
from diplomacy.inference.standard_lstm import load_predictor

TASKS = ('actual_lie', 'suspected_lie')

def read_config(root, task):
    if task not in TASKS: raise ValueError('Invalid task')
    # JSON is a valid subset of YAML, avoiding an unnecessary PyYAML dependency.
    config_path = Path(root) / 'configs' / 'experiments' / 'neural' / 'standard_lstm' / f'{task}.yaml'
    cfg = json.loads(config_path.read_text(encoding='utf8'))
    if cfg['task'] != task: raise ValueError('Config task mismatch')
    return cfg

def processed_paths(root, task):
    return {split: Path(root) / 'artifacts' / 'processed' / task / f'{split}.jsonl'
            for split in ('train', 'validation', 'test')}

def prepare(root, task, cfg=None, device=None, download_glove=False, train_limit=None):
    root = Path(root)
    cfg = cfg or read_config(root, task)
    paths = processed_paths(root, task)
    if not all(p.is_file() for p in paths.values()):
        raise FileNotFoundError('Missing processed files; run Phase 1 pipeline first')
    seed_everything(cfg['seed'])
    rows = {split: load_rows(path, task) for split, path in paths.items()}
    games = {split: {x['game_id'] for x in rec} for split, rec in rows.items()}
    assert not games['train'] & games['validation']
    assert not games['train'] & games['test']
    assert not games['validation'] & games['test']
    if train_limit:
        rows['train'] = rows['train'][:train_limit]
    vocab_dir = root / 'artifacts' / 'vocabularies' / 'standard_lstm' / task
    vocab_path = vocab_dir / 'vocabulary.json'
    checkpoint_path = root / 'artifacts' / 'models' / 'neural' / 'standard_lstm' / task / 'latest_checkpoint.pt'
    if checkpoint_path.exists():
        state = load_checkpoint(checkpoint_path, 'cpu')
        vocab = Vocabulary.from_dict(state['vocabulary'])
        if state['experiment_config'] != cfg:
            raise ValueError('Checkpoint config differs from current config')
    else:
        vocab = Vocabulary.fit((r['message'] for r in rows['train']),
                               min_freq=cfg['data']['min_freq'],
                               max_size=cfg['data']['max_vocab_size'],
                               lowercase=cfg['data']['lowercase'])
    vocab.save(vocab_path)
    data = {split: MessageDataset(records, vocab, cfg['data']['max_tokens'])
            for split, records in rows.items()}
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = {split: DataLoader(dataset, batch_size=cfg['training']['batch_size'],
                                shuffle=(split == 'train'), collate_fn=collate_messages,
                                num_workers=0, pin_memory=device.type == 'cuda')
               for split, dataset in data.items()}
    if checkpoint_path.exists():
        weights = None
        embedding_info = load_checkpoint(checkpoint_path, 'cpu')['embedding_info']
    else:
        weights, embedding_info = initialize_embeddings(
            vocab, dim=cfg['model']['embedding_dim'],
            mode=cfg['embeddings']['mode'],
            glove_zip=root / cfg['embeddings']['glove_zip'],
            download_glove=download_glove or cfg['embeddings']['download_glove'],
            seed=cfg['seed'])
    model = StandardLSTM(len(vocab), embedding_weights=weights, **cfg['model'])
    print(f'{task}: train={len(rows["train"])} val={len(rows["validation"])} '
          f'test={len(rows["test"])} vocab={len(vocab)} device={device}')
    return {'cfg': cfg, 'device': device, 'vocab': vocab, 'model': model,
            'rows': rows, 'loaders': loaders, 'embedding_info': embedding_info,
            'vocabulary_path': vocab_path}

def run_training(root, task, device=None, resume=True, epochs_override=None, download_glove=False,
                 train_limit=None, prepared=None):
    p = prepared if prepared is not None else prepare(root, task, device=device, download_glove=download_glove, train_limit=train_limit)
    folder = Path(root)/'artifacts'/'models'/'neural'/'standard_lstm'/task
    info = train(p['model'], p['loaders']['train'], p['loaders']['validation'],
                 [x['target'] for x in p['rows']['train']], p['cfg'], p['vocab'],
                 p['embedding_info'], task, folder, p['device'], resume, epochs_override)
    info_path = folder / 'training_summary.json'
    info_path.write_text(json.dumps(info, indent=2), encoding='utf8')
    return info

def run_evaluation(root, task, split='validation', device=None):
    if split not in ('validation', 'test'):
        raise ValueError('Split must be validation or test')
    root = Path(root)
    model_dir = root/'artifacts'/'models'/'neural'/'standard_lstm'/task
    checkpoint = model_dir / 'best_model.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(f'No best model for {task}: {checkpoint}')
    state = load_checkpoint(checkpoint, 'cpu')
    cfg = state['experiment_config']
    vocab = Vocabulary.from_dict(state['vocabulary'])
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StandardLSTM(len(vocab), **state['model_config'])
    model.load_state_dict(state['model_state']); model.to(device)
    rows = load_rows(processed_paths(root, task)[split], task)
    dataset = MessageDataset(rows, vocab, cfg['data']['max_tokens'])
    loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'],
                        shuffle=False, collate_fn=collate_messages, num_workers=0)
    train_rows = load_rows(processed_paths(root, task)['train'], task)
    criterion, _ = build_loss([r['target'] for r in train_rows],
                              cfg['training']['class_weight_mode'],
                              cfg['training']['positive_weight'], device)
    results, predictions = evaluate(model, loader, criterion, device)
    results.update({'task': task, 'split': split, 'best_epoch': state['best_epoch'],
                    'checkpoint': str(checkpoint), 'vocab_size': len(vocab),
                    'embedding_info': state['embedding_info']})
    outmetrics = root / 'artifacts'/'metrics'/'standard_lstm'/task
    outpreds = root / 'artifacts'/'predictions'/'standard_lstm'/task
    outmetrics.mkdir(parents=True, exist_ok=True); outpreds.mkdir(parents=True, exist_ok=True)
    (outmetrics/f'{split}_metrics.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    with (outpreds/f'{split}_predictions.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['message_id', 'game_id', 'target', 'prediction', 'prob_lie', 'prob_truth'])
        writer.writeheader(); writer.writerows(predictions)
    print(task, split, 'macro_f1', results['macro_f1'], 'lie_f1', results['lie_f1'])
    return results, predictions

def compare_phase2(root, task):
    root=Path(root)
    phase2 = root/'artifacts'/'metrics'/'traditional'/task/'test_results.csv'
    phase3 = root/'artifacts'/'metrics'/'standard_lstm'/task/'test_metrics.json'
    if not phase2.exists() or not phase3.exists():
        return None
    import pandas as pd
    df = pd.read_csv(phase2)
    selected = ['01_random', '02_majority', '03_harbingers', '04_logistic_regression',
                '06_harbingers_power', '08_bow_lr', '09_bow_power']
    df = df[df['model_id'].isin(selected)][['model_id','accuracy','macro_f1','lie_f1']].copy()
    data = json.loads(phase3.read_text())
    df.loc[len(df)] = ['standard_lstm', data['accuracy'], data['macro_f1'], data['lie_f1']]
    return df

def main():
    parser = argparse.ArgumentParser(description='Phase 3 Standard LSTM')
    parser.add_argument('--root', default='.')
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--action', choices=['train', 'evaluate', 'predict'], default='train')
    parser.add_argument('--split', choices=['validation','test'], default='validation')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--download-glove', action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--text', default='I promise I will help you next season.')
    args = parser.parse_args()
    if args.action == 'train':
        run_training(args.root, args.task, resume=not args.no_resume,
                     download_glove=args.download_glove, epochs_override=args.epochs)
    elif args.action == 'evaluate':
        run_evaluation(args.root, args.task, args.split)
    else:
        path=Path(args.root)/'artifacts'/'models'/'neural'/'standard_lstm'/args.task/'best_model.pt'
        print(load_predictor(path).predict(args.text))

if __name__ == '__main__':
    main()


def save_parity_reference(root, task, messages=None, device=None):
    """Save inference reference values for verifying a Colab checkpoint locally."""
    root = Path(root)
    checkpoint = root/'artifacts'/'models'/'neural'/'standard_lstm'/task/'best_model.pt'
    messages = messages or ['I promise I will help you.', 'Maybe we should talk tomorrow.', '']
    predictor = load_predictor(checkpoint, device=device or 'cpu')
    results = [{'message': message, **predictor.predict(message)} for message in messages]
    output = checkpoint.parent/'parity_reference.json'
    output.write_text(json.dumps(results, indent=2), encoding='utf8')
    return output

def verify_parity_reference(root, task, tolerance=1e-4):
    root = Path(root)
    folder = root/'artifacts'/'models'/'neural'/'standard_lstm'/task
    reference = json.loads((folder/'parity_reference.json').read_text(encoding='utf8'))
    predictor = load_predictor(folder/'best_model.pt', device='cpu')
    for row in reference:
        current = predictor.predict(row['message'])
        assert current['prediction'] == row['prediction'], 'Prediction mismatch'
        assert abs(current['prob_lie']-row['prob_lie']) <= tolerance, 'Probability mismatch'
    return {'task':task, 'checked':len(reference), 'tolerance':tolerance, 'passed': True}

def write_error_analysis(root, task, split='test', limit=30):
    import pandas as pd
    root = Path(root)
    pred_path=root/'artifacts'/'predictions'/'standard_lstm'/task/f'{split}_predictions.csv'
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    df=pd.read_csv(pred_path)
    mismatches=df[df['target'] != df['prediction']].copy()
    text_map = {r['message_id']:r['message'] for r in load_rows(processed_paths(root,task)[split],task)}
    mismatches['message']=mismatches['message_id'].map(text_map)
    folder=root/'artifacts'/'reports'/'phase_3'
    folder.mkdir(parents=True,exist_ok=True)
    output=folder/f'{task}_{split}_errors.csv'
    mismatches.head(limit).to_csv(output,index=False)
    return output
