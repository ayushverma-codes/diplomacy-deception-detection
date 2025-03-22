"""Ten explicitly named baseline experiments for each of two annotation tasks.

Published paper uses LR for Harbingers, BoW, and +power. The duplicated
model-family entries are kept as separately tracked named experiments.
"""
import argparse, hashlib, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, issparse
from sklearn.preprocessing import StandardScaler
from diplomacy.features.harbingers_features import HarbingersFeatures
from diplomacy.features.numerical_features import NumericalFeatures
from diplomacy.features.bow_features import make_bow
from diplomacy.features.power_features import extract_power, fit_power_imputer
from diplomacy.features.feature_combiner import combine
from diplomacy.models.baselines.random_baseline import random_predict
from diplomacy.models.baselines.majority_baseline import majority_predict
from diplomacy.training.baseline_trainer import select_logistic
from diplomacy.evaluation.metrics import calculate_metrics
from diplomacy.evaluation.confusion_matrix import save_confusion

SPECS = [
 ('01_random','none','random'),
 ('02_majority','none','majority'),
 ('03_harbingers','harbingers','logistic_regression'),
 ('04_logistic_regression','numerical','logistic_regression'),
 ('05_harbingers_lr','harbingers','logistic_regression'),
 ('06_harbingers_power','harbingers+power','logistic_regression'),
 ('07_harbingers_lr_power','harbingers+power','logistic_regression'),
 ('08_bow_lr','bow','logistic_regression'),
 ('09_bow_power','bow+power','logistic_regression'),
 ('10_bow_lr_power','bow+power','logistic_regression'),
]
SPLITS=('train','validation','test')
TASKS=('actual_lie','suspected_lie')

def load_rows(root,task,split):
    path=Path(root)/'artifacts'/'processed'/task/f'{split}.jsonl'
    with path.open(encoding='utf8') as stream: rows=[json.loads(line) for line in stream if line.strip()]
    assert all(x['target'] in (0,1) for x in rows)
    assert len({x['message_id'] for x in rows}) == len(rows), 'Duplicate message IDs'
    return rows

def prepare_features(rows_by_split,root,task):
    texts={split:[r.get('normalized_message') or r['message'] or '' for r in rows] for split,rows in rows_by_split.items()}
    output={}
    extractor=HarbingersFeatures()
    output['harbingers']={s:extractor.transform(texts[s]) for s in SPLITS}
    joblib.dump(extractor,root/'artifacts/features/harbingers'/f'{task}_extractor.joblib')
    numerical=NumericalFeatures()
    numeric={s:numerical.transform(texts[s]) for s in SPLITS}
    numeric_scaler=StandardScaler().fit(numeric['train'])
    output['numerical']={s:numeric_scaler.transform(numeric[s]) for s in SPLITS}
    joblib.dump({'extractor':numerical,'scaler':numeric_scaler},root/'artifacts/features/numerical'/f'{task}_extractor.joblib')
    bow=make_bow()
    output['bow']={s:bow.fit_transform(texts[s]) if s=='train' else bow.transform(texts[s]) for s in SPLITS}
    joblib.dump(bow,root/'artifacts/features/bow'/f'{task}_vectorizer.joblib')
    power_raw={s:extract_power(rows_by_split[s]) for s in SPLITS}
    imputer=fit_power_imputer(power_raw['train'])
    power={s:imputer.transform(power_raw[s]) for s in SPLITS}
    scaler=StandardScaler().fit(power['train'])
    power={s:scaler.transform(power[s]) for s in SPLITS}
    joblib.dump({'imputer':imputer,'scaler':scaler},root/'artifacts/features/power'/f'{task}_transformer.joblib')
    output['harbingers+power']={s:combine(output['harbingers'][s],power[s]) for s in SPLITS}
    output['bow+power']={s:combine(output['bow'][s],power[s]) for s in SPLITS}
    for name,values in output.items():
        for split,mat in values.items():
            assert mat.shape[0]==len(rows_by_split[split])
    # Persist the feature matrices so notebooks and later experiments can reuse them.
    for name, values in output.items():
        folder=root/'artifacts/features'/name.replace('+','_')/task
        folder.mkdir(parents=True,exist_ok=True)
        for split, matrix in values.items():
            if issparse(matrix):
                save_npz(folder/f'{split}.npz',matrix)
            else:
                np.save(folder/f'{split}.npy',matrix)
    info={'vocabulary_size':len(bow.vocabulary_),
          'dimensions':{name:values['train'].shape[1] for name,values in output.items()},
          'subjectivity_lexicon_complete':False}
    (root/'artifacts/features'/f'{task}_info.json').write_text(json.dumps(info,indent=2))
    return output,info

def serialize_predictions(path,rows,pred,scores):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8') as f:
        for row,p,s in zip(rows,pred,scores):
            f.write(json.dumps({'message_id':row['message_id'],'game_id':row['game_id'],
              'conversation_index':row['conversation_index'],'message_position':row['message_position'],
              'target':row['target'],'prediction':int(p),'score':float(s)},ensure_ascii=False)+'\n')

def execute(root, tasks=TASKS, seed=42, quick=False):
    root=Path(root).resolve()
    assert sorted(p.name for p in (root/'data').iterdir())==['test.jsonl','train.jsonl','validation.jsonl'], 'data/ must have only original JSONL files'
    (root/'artifacts/features/harbingers').mkdir(parents=True,exist_ok=True)
    for task in tasks:
        for sub in ['features/bow','features/power','features/numerical',
                    f'models/traditional/{task}',f'predictions/traditional/{task}',
                    f'metrics/traditional/{task}','figures/traditional','logs/traditional']:
            (root/'artifacts'/sub).mkdir(parents=True,exist_ok=True)
        rows={s:load_rows(root,task,s) for s in SPLITS}
        y={s:np.asarray([r['target'] for r in rows[s]],dtype=int) for s in SPLITS}
        assert len(set(y['train']))==2
        features,info=prepare_features(rows,root,task)
        all_results=[]
        for model_id,feature,classifier in SPECS:
            cfg_path=root/'configs/experiments/traditional'/f'{model_id[3:]}.yaml'
            config=json.loads(cfg_path.read_text())
            grid=config['search']
            if quick:grid={'C':[1.0],'class_weight':[None]}
            base={'model_id':model_id,'model_name':model_id[3:],
                  'task':task,'feature_representation':feature,'classifier':classifier,
                  'random_seed':seed,'class_distribution':np.bincount(y['train'],minlength=2).tolist(),
                  'training_count':len(y['train']),'validation_count':len(y['validation']),'test_count':len(y['test'])}
            if classifier=='random':
                predictions={s:random_predict(y['train'],len(y[s]),seed=seed+(0 if s=='validation' else 100)) for s in ['validation','test']}
                params={'train_positive_rate':float(y['train'].mean()),'seed':seed}
            elif classifier=='majority':
                predictions={}
                for s in ['validation','test']:
                    pred,proba,label=majority_predict(y['train'],len(y[s]))
                    predictions[s]=(pred,proba)
                params={'majority_class':label}
            else:
                selected=select_logistic(features[feature]['train'],y['train'],features[feature]['validation'],y['validation'],grid,seed)
                search_log=root/'artifacts/logs/traditional'/f'{task}_{model_id}_search.json'
                search_log.write_text(json.dumps(selected['history'],indent=2)+'\n')
                model=selected['model'];params={'C':selected['C'],'class_weight':selected['class_weight'],'seed':seed}
                joblib.dump({'model':model,'feature_representation':feature,'parameters':params},root/'artifacts/models/traditional'/task/f'{model_id}.joblib')
                predictions={s:(model.predict(features[feature][s]),model.predict_proba(features[feature][s])[:,1]) for s in ['validation','test']}
            for s,(pred,proba) in predictions.items():
                metrics=calculate_metrics(y[s],pred,None if classifier=='random' else proba)
                result={**base,'split':s,'parameters':params,'run_id':f'{task}_{model_id}_seed{seed}',**metrics}
                all_results.append(result)
                serialize_predictions(root/'artifacts/predictions/traditional'/task/f'{model_id}_{s}.jsonl',rows[s],pred,proba)
                save_confusion(metrics['confusion_matrix'],root/'artifacts/figures/traditional'/f'{task}_{model_id}_{s}.png',f'{task}: {model_id} ({s})')
            print(task,model_id,'validation',round(all_results[-2]['macro_f1'],4),'test',round(all_results[-1]['macro_f1'],4),flush=True)
        metrics_path=root/'artifacts/metrics/traditional'/task
        (metrics_path/'detailed_metrics.json').write_text(json.dumps(all_results,indent=2,ensure_ascii=False)+'\n')
        for split in ['validation','test']:
            records=[r for r in all_results if r['split']==split]
            pd.DataFrame(records).to_csv(metrics_path/f'{split}_results.csv',index=False)
        print('Completed task:',task,flush=True)
    assert sorted(p.name for p in (root/'data').iterdir())==['test.jsonl','train.jsonl','validation.jsonl']

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='.')
    parser.add_argument('--quick',action='store_true',help='Smoke-test grid; not full hyperparameter search')
    parser.add_argument('--task',choices=TASKS,action='append')
    args=parser.parse_args()
    execute(args.root,tuple(args.task) if args.task else TASKS,quick=args.quick)
