"""Reusable deterministic training, metrics, checkpointing and evaluation."""
from __future__ import annotations
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score)

from diplomacy.models.neural.lstm import StandardLSTM
from diplomacy.data.text_dataset import Vocabulary


def seed_everything(seed=1994):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loss(labels, weighting='paper', positive_weight=30.0, device='cpu'):
    if weighting == 'paper':
        weights = [1.0, float(positive_weight)]
    elif weighting == 'balanced':
        negatives = sum(x == 0 for x in labels)
        positives = sum(x == 1 for x in labels)
        n = len(labels)
        weights = [n / (2 * max(negatives, 1)), n / (2 * max(positives, 1))]
    elif weighting == 'none':
        weights = [1., 1.]
    else:
        raise ValueError('Invalid class weighting')
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device)), weights


def metrics(y_true, y_pred, probabilities):
    scores = {'accuracy': float(accuracy_score(y_true, y_pred)),
              'precision': float(precision_score(y_true, y_pred, zero_division=0)),
              'recall': float(recall_score(y_true, y_pred, zero_division=0)),
              'lie_f1': float(f1_score(y_true, y_pred, zero_division=0)),
              'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
              'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
              'n_samples': len(y_true)}
    scores['pr_auc'] = (float(average_precision_score(y_true, probabilities))
                        if 1 in y_true else None)
    return scores

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    targets, predicted, probabilities, predictions = [], [], [], []
    total_loss, loss_weight_total, sample_count = 0., 0., 0
    for batch in loader:
        logits = model(batch['input_ids'].to(device), batch['lengths'].to(device))
        labels = batch['targets'].to(device)
        # Weighted CE's mean reduction normalizes by sum of class weights; compute
        # sample-level loss consistently by using unreduced weighted CE.
        element_loss = nn.functional.cross_entropy(logits, labels, weight=criterion.weight, reduction='none')
        total_loss += element_loss.sum().item()
        loss_weight_total += criterion.weight[labels].sum().item()
        sample_count += len(labels)
        probs = logits.softmax(dim=1)[:, 1].cpu().tolist()
        guesses = logits.argmax(dim=1).cpu().tolist()
        actual = labels.cpu().tolist()
        targets.extend(actual); predicted.extend(guesses); probabilities.extend(probs)
        for i, mid in enumerate(batch['message_ids']):
            predictions.append({'message_id': mid, 'game_id': batch['game_ids'][i],
                                'target': actual[i], 'prediction': guesses[i],
                                'prob_lie': probs[i], 'prob_truth': 1.0 - probs[i]})
    score = metrics(targets, predicted, probabilities)
    score['loss'] = total_loss / max(loss_weight_total, 1e-12)
    return score, predictions


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    if not state:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    torch.set_rng_state(state["torch"].cpu())

    if state.get("cuda") is not None and torch.cuda.is_available():
        cuda_states = [
            s.detach().to(device="cpu", dtype=torch.uint8)
            for s in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def save_checkpoint(path, model, optimizer, epoch, history, best_score, best_epoch,
                    model_config, vocab, experiment_config, embedding_info, task, seed,
                    scheduler=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
               'scheduler_state': scheduler.state_dict() if scheduler else None,
               'epoch': epoch, 'history': history, 'best_score': best_score,
               'best_epoch': best_epoch, 'model_config': model_config,
               'vocabulary': vocab.to_dict(), 'experiment_config': experiment_config,
               'embedding_info': embedding_info, 'class_mapping': {'truth': 0, 'lie': 1},
               'task': task, 'seed': seed, 'rng_state': rng_state(),
               'versions': {'pytorch': torch.__version__, 'numpy': np.__version__}}
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, map_location='cpu', trusted=True):
    # Only load checkpoints produced by this project, not untrusted third-party files.
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        if not trusted:
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


def train(model, train_loader, val_loader, train_labels, cfg, vocab, embedding_info,
          task, output_dir, device, resume=True, epochs_override=None):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = cfg['training']; seed = cfg['seed']
    criterion, weights = build_loss(train_labels, train_cfg['class_weight_mode'],
                                    train_cfg['positive_weight'], device)
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=float(train_cfg['learning_rate']))
    max_epochs = int(epochs_override or train_cfg['epochs'])
    patience = int(train_cfg['patience'])
    best_path = output_dir / 'best_model.pt'
    last_path = output_dir / 'latest_checkpoint.pt'
    start = 0; best_score = float('-inf'); best_epoch = 0; history = []
    if resume and last_path.exists():
        state = load_checkpoint(last_path, device)
        if state['task'] != task or state['model_config'] != cfg['model'] or state['vocabulary'] != vocab.to_dict():
            raise ValueError('Resume checkpoint configuration/vocabulary mismatch')
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        start = state['epoch']; best_score = state['best_score']
        best_epoch = state['best_epoch']; history = state['history']
        restore_rng(state.get('rng_state'))
        print(f'Resuming {task} from completed epoch {start}')
    start_time = time.time()
    for epoch in range(start, max_epochs):
        if epoch - best_epoch >= patience:
            print('Early stopping: patience exceeded')
            break
        model.train(); loss_sum = 0.; loss_weight_total = 0.; count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch['input_ids'].to(device), batch['lengths'].to(device))
            loss = criterion(logits, batch['targets'].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg['gradient_clip']))
            optimizer.step()
            weight_total = criterion.weight[batch['targets'].to(device)].sum().item()
            loss_sum += loss.item() * weight_total
            loss_weight_total += weight_total
            count += len(batch['targets'])
        val_score, _ = evaluate(model, val_loader, criterion, device)
        row = {'epoch': epoch + 1, 'train_loss': loss_sum / max(1e-12, loss_weight_total),
               'validation_loss': val_score['loss'], 'validation_macro_f1': val_score['macro_f1'],
               'validation_lie_f1': val_score['lie_f1']}
        history.append(row)
        is_better = val_score['macro_f1'] > best_score
        if is_better:
            best_score = val_score['macro_f1']; best_epoch = epoch + 1
            save_checkpoint(best_path, model, optimizer, epoch + 1, history, best_score,
                            best_epoch, cfg['model'], vocab, cfg, embedding_info, task, seed)
        save_checkpoint(last_path, model, optimizer, epoch + 1, history, best_score,
                        best_epoch, cfg['model'], vocab, cfg, embedding_info, task, seed)
        print(f'{task} epoch {epoch + 1}/{max_epochs} train_loss={row["train_loss"]:.4f}'
              f' val_macro_f1={val_score["macro_f1"]:.4f}'
              f' val_lie_f1={val_score["lie_f1"]:.4f}')
    (output_dir / 'history.json').write_text(json.dumps(history, indent=2))
    return {'history': history, 'best_epoch': best_epoch, 'best_val_macro_f1': best_score,
            'duration_seconds_this_session': time.time()-start_time, 'class_weights': weights}
