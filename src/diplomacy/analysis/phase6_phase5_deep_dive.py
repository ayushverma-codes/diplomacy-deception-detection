"""Phase 6: verify Phase 5 test predictions, evaluate validation-only threshold diagnostics.

Usage: python -m diplomacy.analysis.phase6_phase5_deep_dive --root .
No training, no edits to original prediction/checkpoint files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, accuracy_score, roc_auc_score)

ARCHS = ('hierarchical_lstm', 'hierarchical_lstm_power', 'hierarchical_lstm_bert', 'hierarchical_lstm_bert_power')
TASKS = ('actual_lie', 'suspected_lie')
KEYS = ['game_id', 'message_id', 'absolute_message_index']


def measure(y, pred):
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', labels=[0, 1], zero_division=0)),
        'lie_f1': float(f1_score(y, pred, pos_label=1, zero_division=0)),
        'lie_precision': float(precision_score(y, pred, pos_label=1, zero_division=0)),
        'lie_recall': float(recall_score(y, pred, pos_label=1, zero_division=0)),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).astype(int).tolist(),
        'predicted_positive': int(np.sum(pred)),
        'positives': int(np.sum(y)),
        'samples': int(len(y)),
    }


def load_split(root, arch, task, split):
    p = root / 'artifacts/phase_5/predictions' / arch / task / f'{split}_predictions.csv'
    if not p.exists():
        raise FileNotFoundError(f'Missing prediction file: {p}')
    df = pd.read_csv(p, dtype={'game_id': str, 'message_id': str})
    required = KEYS + ['target', 'prediction', 'prob_lie']
    if any(col not in df.columns for col in required):
        raise ValueError(f'{p}: missing fields {set(required) - set(df.columns)}')
    if df.duplicated(KEYS).any():
        raise ValueError(f'{p}: duplicate message identifiers')
    if len(df) == 0 or df[required].isna().any().any():
        raise ValueError(f'{p}: missing values or empty split')
    if not df.target.isin([0, 1]).all() or not df.prediction.isin([0, 1]).all():
        raise ValueError(f'{p}: non-binary target/prediction')
    if not df.prob_lie.between(0, 1).all():
        raise ValueError(f'{p}: invalid probabilities')
    return df


def select_threshold(validation, metric='lie_f1'):
    """Fixed candidate grid chosen on validation only, never using test labels."""
    candidates = np.round(np.arange(.01, .501, .01), 3)
    y = validation.target.to_numpy(dtype=int)
    p = validation.prob_lie.to_numpy(dtype=float)
    scored = [(float(t), measure(y, (p >= t).astype(int))[metric]) for t in candidates]
    maximum = max(score for _, score in scored)
    # Tie prefers higher threshold, reducing arbitrary extra predicted positives.
    threshold = max(t for t, score in scored if np.isclose(score, maximum))
    return threshold, maximum


def probability_summary(df):
    p = df.prob_lie.to_numpy(dtype=float)
    y = df.target.to_numpy(dtype=int)
    result = {
        'min': float(np.min(p)), 'max': float(np.max(p)),
        'mean': float(np.mean(p)), 'median': float(np.median(p)),
        'std': float(np.std(p)),
        'mean_positive': float(np.mean(p[y == 1])) if any(y == 1) else None,
        'mean_negative': float(np.mean(p[y == 0])) if any(y == 0) else None,
        'average_precision': float(average_precision_score(y, p)),
        'roc_auc': float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        'brier_score': float(brier_score_loss(y, p)),
    }
    return result


def validate_metrics(df, root, arch, task, split):
    mpath = root / 'artifacts/phase_5/metrics' / arch / task / f'{split}_metrics.json'
    if not mpath.exists():
        raise FileNotFoundError(mpath)
    saved = json.loads(mpath.read_text(encoding='utf-8'))
    observed = measure(df.target.to_numpy(int), df.prediction.to_numpy(int))
    for k in ['accuracy', 'macro_f1', 'lie_f1']:
        if not np.isclose(observed[k], float(saved[k]), atol=1e-9):
            raise ValueError(f'{arch}/{task}/{split}: {k} mismatch saved={saved[k]} computed={observed[k]}')
    if observed['confusion_matrix'] != saved['confusion_matrix']:
        raise ValueError(f'{arch}/{task}/{split}: confusion matrix mismatch')
    return observed


def aligned_frame(dfs, split, task):
    # Strictly equal message populations and ground-truth labels across architectures.
    base = dfs[(ARCHS[0], task, split)][KEYS + ['target']].copy()
    for arch in ARCHS[1:]:
        cur = dfs[(arch, task, split)]
        merged = base.merge(cur[KEYS + ['target']], on=KEYS, suffixes=('_base', '_current'), how='outer', indicator=True, validate='one_to_one')
        if (merged['_merge'] != 'both').any() or (merged.target_base != merged.target_current).any():
            raise ValueError(f'{task}/{split}: population/target mismatch for {arch}')
    return base


def run(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = output / 'bert_diagnostics'; diagnostic_dir.mkdir(exist_ok=True)
    error_dir = output / 'error_analysis'; error_dir.mkdir(exist_ok=True)
    ablation_dir = output / 'ablation'; ablation_dir.mkdir(exist_ok=True)
    plot_dir = output / 'plots'; plot_dir.mkdir(exist_ok=True)

    dfs, aggregate, probrows, thresholdrows, games, logrows = {}, [], [], [], [], []
    for task in TASKS:
        for arch in ARCHS:
            for split in ['validation', 'test']:
                df = load_split(root, arch, task, split)
                dfs[(arch, task, split)] = df
                observed = validate_metrics(df, root, arch, task, split)
                aggregate.append({'architecture':arch, 'task':task, 'split':split, **{k:v for k,v in observed.items() if k!='confusion_matrix'}, 'confusion_matrix':json.dumps(observed['confusion_matrix'])})
                probrows.append({'architecture':arch, 'task':task, 'split':split, **probability_summary(df)})
                if split == 'test':
                    for game, part in df.groupby('game_id'):
                        mm=measure(part.target.to_numpy(int),part.prediction.to_numpy(int))
                        games.append({'architecture':arch,'task':task,'game_id':game,**{k:v for k,v in mm.items() if k!='confusion_matrix'}})
            val, test = dfs[(arch,task,'validation')],dfs[(arch,task,'test')]
            th, val_best = select_threshold(val)
            new_val=measure(val.target.to_numpy(int), (val.prob_lie.to_numpy() >= th).astype(int))
            new_test=measure(test.target.to_numpy(int), (test.prob_lie.to_numpy() >= th).astype(int))
            thresholdrows.append({'architecture':arch,'task':task,'validation_selected_threshold':th,'selection_metric':'positive_class_f1',
                'validation_selected_positive_f1':val_best, 'validation_selected_macro_f1':new_val['macro_f1'],
                'test_original_positive_f1':measure(test.target.to_numpy(int),test.prediction.to_numpy(int))['lie_f1'],
                'test_adjusted_positive_f1':new_test['lie_f1'], 'test_adjusted_macro_f1':new_test['macro_f1'],
                'test_adjusted_precision':new_test['lie_precision'],'test_adjusted_recall':new_test['lie_recall'],
                'test_adjusted_predicted_positive':new_test['predicted_positive'],
                'test_adjusted_confusion_matrix':json.dumps(new_test['confusion_matrix'])})

            logs = root / 'artifacts/phase_5/logs' / arch / task / 'epochs.jsonl'
            if not logs.exists():
                raise FileNotFoundError(logs)
            lines=[json.loads(line) for line in logs.read_text(encoding='utf-8').splitlines() if line.strip()]
            if not lines:raise ValueError(f'Empty training log: {logs}')
            summary_path=root / 'artifacts/phase_5/checkpoints' / arch / task / 'training_summary.json'
            summary=json.loads(summary_path.read_text(encoding='utf-8'))
            for item in lines:
                logrows.append({'architecture':arch,'task':task,**item,'selected_best_epoch':summary.get('best_epoch')})

    for task in TASKS:
        for split in ('validation','test'):
            aligned_frame(dfs,split,task)

    agg=pd.DataFrame(aggregate); probs=pd.DataFrame(probrows); ths=pd.DataFrame(thresholdrows); pergame=pd.DataFrame(games); logs=pd.DataFrame(logrows)
    agg.to_csv(ablation_dir/'phase5_prediction_metrics_recomputed.csv',index=False)
    probs.to_csv(diagnostic_dir/'probability_distributions.csv',index=False)
    ths.to_csv(diagnostic_dir/'validation_selected_thresholds.csv',index=False)
    logs.to_csv(diagnostic_dir/'epoch_training_history.csv',index=False)
    pergame.to_csv(error_dir/'phase5_per_game_metrics.csv',index=False)

    # Pairwise, strictly aligned predictions and errors with stable identifiers.
    disag=[]; case_rows=[]
    for task in TASKS:
        base=dfs[(ARCHS[0],task,'test')].set_index(KEYS)
        for arch in ARCHS:
            cur=dfs[(arch,task,'test')].set_index(KEYS)
            joined=base[['target','prediction','prob_lie']].join(cur[['prediction','prob_lie']],lsuffix='_base',rsuffix='_model',validate='one_to_one')
            for key, row in joined.iterrows():
                target=int(row.target)
                pred=int(row.prediction_model)
                if arch != ARCHS[0] and int(row.prediction_base)!=pred:
                    disag.append({'architecture':arch,'task':task,'game_id':key[0],'message_id':key[1],'absolute_message_index':key[2],
                                  'target':target,'baseline_prediction':int(row.prediction_base),'model_prediction':pred,
                                  'baseline_prob_lie':row.prob_lie_base,'model_prob_lie':row.prob_lie_model})
                if target!=pred:
                    case_rows.append({'architecture':arch,'task':task,'game_id':key[0],'message_id':key[1],
                                      'absolute_message_index':key[2],'target':target,'prediction':pred,'prob_lie':row.prob_lie_model,
                                      'error_type':'false_negative' if target else 'false_positive'})
    pd.DataFrame(disag).to_csv(error_dir/'phase5_disagreements_vs_baseline.csv',index=False)
    pd.DataFrame(case_rows).to_csv(error_dir/'phase5_all_errors.csv',index=False)

    ablations=[]
    for task in TASKS:
        df=agg[(agg.task==task)&(agg.split=='test')].set_index('architecture')
        for col in ('macro_f1','lie_f1','lie_precision','lie_recall'):
            b,p,br,bp=[float(df.loc[a,col]) for a in ARCHS]
            ablations.append({'task':task,'metric':col,'power_change':p-b,'bert_change':br-b,
                              'bert_power_vs_bert':bp-br,'interaction':bp-br-p+b})
    pd.DataFrame(ablations).to_csv(ablation_dir/'phase5_verified_ablation_deltas.csv',index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for task in TASKS:
        fig, ax=plt.subplots(figsize=(7,4))
        for arch in ARCHS:
            df=dfs[(arch,task,'test')]
            ax.hist(df.prob_lie,bins=np.linspace(0,1,41),histtype='step',density=True,label=arch)
        ax.set(xlabel='Test predicted probability of lie',ylabel='Density',title=f'{task}: score distributions')
        ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(plot_dir/f'{task}_phase5_probability_histograms.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4))
        for arch in ('hierarchical_lstm_bert','hierarchical_lstm_bert_power'):
            ds=dfs[(arch,task,'test')]
            ax.hist(ds.loc[ds.target==0,'prob_lie'],bins=30,alpha=.4,density=True,label=f'{arch} negative')
            ax.hist(ds.loc[ds.target==1,'prob_lie'],bins=30,alpha=.4,density=True,label=f'{arch} positive')
        ax.set(xlabel='Predicted probability of lie',ylabel='Density',title=f'{task}: BERT class separation')
        ax.legend(fontsize=6);fig.tight_layout();fig.savefig(plot_dir/f'{task}_bert_class_distributions.png',dpi=160);plt.close(fig)
    report={
        'provenance':'Phase 5 user-supplied prediction CSVs, metrics JSON, training logs and summaries; no checkpoint execution',
        'experiments_validated':len(agg[agg.split=='test']),
        'prediction_alignment':'all four models have identical game/message/index and ground truth within each task and split',
        'phase5_test_rows':agg[agg.split=='test'].to_dict('records'),
        'probability_statistics':probs.to_dict('records'),
        'threshold_analysis':ths.to_dict('records'),
        'training_epochs':logs.groupby(['architecture','task']).size().rename('n').reset_index().to_dict('records'),
        'limitations':['Test set contains only two games; no game-level significance claim', 'Threshold candidates selected on single-game validation and evaluated once on test','Source CSV contains no text, sender/receiver, season or power delta; cannot infer human intent','BERT gradient/representation dynamics are not proven from scalar logs']
    }
    (diagnostic_dir/'phase5_deep_dive_summary.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print('Verified:',len(agg[agg.split=='test']),'test predictions/metrics; four task/split populations aligned across all architectures')
    print(ths[['architecture','task','validation_selected_threshold','test_adjusted_positive_f1','test_adjusted_macro_f1']].to_string(index=False))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    root=args.root.resolve()
    out=(args.output if args.output else root/'artifacts/phase_6').resolve()
    run(root,out)

if __name__=='__main__':main()
