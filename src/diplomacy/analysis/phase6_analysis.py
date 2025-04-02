"""Phase 6 artifact-only audit and ablation (no model execution, no retraining).
Usage: python run_phase6_analysis.py --root PATH --output PATH
Depends on: Python stdlib, pandas, numpy, matplotlib.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARCHS=['hierarchical_lstm','hierarchical_lstm_power','hierarchical_lstm_bert','hierarchical_lstm_bert_power']
TASKS=['actual_lie','suspected_lie']
PRED_TEMPLATES={
 'Phase 3 Standard LSTM':'artifacts/predictions/standard_lstm/{task}/test_predictions.csv',
 'Phase 4 BiLSTM':'artifacts/predictions/phase_4/bilstm/{task}/test_predictions.csv',
 'Phase 4 concatenated':'artifacts/predictions/phase_4/bilstm_concat_context/{task}/test_predictions.csv',
 'Phase 4 hierarchical':'artifacts/predictions/phase_4/hierarchical_lstm/{task}/test_predictions.csv',
 'Phase 2 BoW + Power':'artifacts/predictions/traditional/{task}/09_bow_power_test.jsonl'
}

def dump(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(data,indent=2,allow_nan=False,default=str),encoding='utf-8')

def read_json(p):return json.loads(p.read_text(encoding='utf-8'))
def load_pred(path):
 if path.suffix=='.jsonl':
  d=pd.DataFrame([json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]); d['prob_lie']=d.get('score',np.nan)
 else:d=pd.read_csv(path)
 required={'message_id','game_id','target','prediction'}
 if not required.issubset(d.columns):raise ValueError(f'Missing prediction fields {required-set(d.columns)}: {path}')
 if d.message_id.duplicated().any():raise ValueError(f'Duplicate message_id in {path}')
 return d

def metric_from_cm(cm):
 tn,fp=cm[0];fn,tp=cm[1];pp=tp/(tp+fp) if tp+fp else 0;rr=tp/(tp+fn) if tp+fn else 0;f1=2*pp*rr/(pp+rr) if pp+rr else 0
 p0=tn/(tn+fn) if tn+fn else 0;r0=tn/(tn+fp) if tn+fp else 0;f0=2*p0*r0/(p0+r0) if p0+r0 else 0
 return {'accuracy':(tn+tp)/sum(map(sum,cm)),'precision':pp,'recall':rr,'lie_f1':f1,'macro_f1':(f0+f1)/2}

def generate(root,out):
 out.mkdir(parents=True,exist_ok=True)
 for sub in ('audit','ablation','error_analysis','interpretability','benchmarking'):
  (out/sub).mkdir(parents=True,exist_ok=True)
 notes=[]; rows=[]; valid=0
 for a in ARCHS:
  for task in TASKS:
   p=root/f'artifacts/phase_5/metrics/{a}/{task}/test_metrics.json'
   if not p.exists():notes.append(f'MISSING Phase 5 test metrics: {a}/{task}');continue
   m=read_json(p);r=metric_from_cm(m['confusion_matrix'])
   diff={k:abs(float(m[k])-v) for k,v in r.items()}
   if any(v>1e-7 for v in diff.values()):notes.append(f'METRIC INCONSISTENCY {a}/{task} {diff}')
   else:valid+=1
   tn,fp=m['confusion_matrix'][0];fn,tp=m['confusion_matrix'][1]
   rows.append({'architecture':a,'task':task,'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'precision':m['precision'],'recall':m['recall'],'lie_f1':m['lie_f1'],'pr_auc':m.get('pr_auc'),'n_samples':m['n_samples'],'tn':tn,'fp':fp,'fn':fn,'tp':tp,'test_metrics_path':str(p.relative_to(root)),'best_epoch':m.get('best_epoch')})
 df=pd.DataFrame(rows);(out/'ablation').mkdir(exist_ok=True);df.to_csv(out/'ablation/phase5_test_metrics_verified.csv',index=False)
 deltas=[]
 for task in TASKS:
  d=df.set_index(['task','architecture'])
  if not all((task,a) in d.index for a in ARCHS):continue
  base=d.loc[(task,ARCHS[0])];pw=d.loc[(task,ARCHS[1])];bt=d.loc[(task,ARCHS[2])];both=d.loc[(task,ARCHS[3])]
  for feature,compare in [('power',pw),('bert',bt),('bert_plus_power',both)]:
   for met in ['accuracy','macro_f1','precision','recall','lie_f1']:
    v=float(compare[met]);b=float(base[met]);deltas.append({'task':task,'comparison':feature,'metric':met,'baseline':b,'variant':v,'absolute_change':v-b,'relative_change_percent':(v-b)/b*100 if b else None})
  for met in ['accuracy','macro_f1','lie_f1']:
   deltas.append({'task':task,'comparison':'interaction','metric':met,'baseline':float(base[met]),'variant':float(both[met]),'absolute_change':float(both[met]-bt[met]-pw[met]+base[met]),'relative_change_percent':None})
 pd.DataFrame(deltas).to_csv(out/'ablation/power_bert_ablation.csv',index=False)
 audit=[];historic=[];disagree=[];pergame=[];calibration=[]
 for task in TASKS:
  preds={}
  for name,tpl in PRED_TEMPLATES.items():
   p=root/tpl.format(task=task)
   if not p.exists():audit.append({'task':task,'model':name,'status':'missing','path':str(p.relative_to(root))});continue
   try:
    x=load_pred(p);preds[name]=x;audit.append({'task':task,'model':name,'status':'present','count':len(x),'unique_ids':x.message_id.nunique(),'path':str(p.relative_to(root))})
    tn=((x.target==0)&(x.prediction==0)).sum();fp=((x.target==0)&(x.prediction==1)).sum();fn=((x.target==1)&(x.prediction==0)).sum();tp=((x.target==1)&(x.prediction==1)).sum()
    hist={'task':task,'model':name,'n':len(x),**metric_from_cm([[int(tn),int(fp)],[int(fn),int(tp)]])}
    historic.append(hist)
    for g,gdf in x.groupby('game_id'):
     tn=((gdf.target==0)&(gdf.prediction==0)).sum();fp=((gdf.target==0)&(gdf.prediction==1)).sum();fn=((gdf.target==1)&(gdf.prediction==0)).sum();tp=((gdf.target==1)&(gdf.prediction==1)).sum()
     pergame.append({'model':name,'task':task,'game_id':g,'n':len(gdf),'positives':int(gdf.target.sum()),**metric_from_cm([[int(tn),int(fp)],[int(fn),int(tp)]])})
    if 'prob_lie' in x:
     prob=pd.to_numeric(x.prob_lie,errors='coerce');y=x.target.astype(int);edges=np.linspace(0,1,11)
     for i in range(10):
      ix=(prob>=edges[i])&(prob<(edges[i+1] if i!=9 else 1.000001));n=int(ix.sum())
      if n:calibration.append({'task':task,'model':name,'bin':i,'count':n,'avg_probability':prob[ix].mean(),'positive_rate':y[ix].mean()})
   except Exception as e:audit.append({'task':task,'model':name,'status':f'ERROR {str(e)}'})
  if 'Phase 4 hierarchical' in preds and 'Phase 4 BiLSTM' in preds:
   a=preds['Phase 4 hierarchical'];b=preds['Phase 4 BiLSTM'];joined=a.merge(b,on=['message_id','game_id'],suffixes=('_hier','_bilstm'),validate='one_to_one')
   if not (joined.target_hier==joined.target_bilstm).all():notes.append(f'Phase 4 comparison inconsistent target labels for {task}')
   else:
    ds=joined[joined.prediction_hier!=joined.prediction_bilstm].copy(); ds['task']=task
    disagree.extend(ds.to_dict(orient='records'))
 for task in TASKS:
  for a in ARCHS:
   pp=root/f'artifacts/phase_5/predictions/{a}/{task}/test_predictions.csv'
   hh=root/f'artifacts/phase_5/logs/{a}/{task}/epochs.jsonl'
   audit.append({'task':task,'model':'Phase 5 '+a,'status':'predictions present' if pp.exists() else 'per-message predictions missing','path':str(pp.relative_to(root))})
   if not hh.exists():notes.append(f'Phase 5 training history not supplied: {a}/{task}')
 pd.DataFrame(audit).to_csv(out/'audit/artifact_inventory.csv',index=False)
 pd.DataFrame(historic).to_csv(out/'benchmarking/phase2_to_phase4_prediction_metrics.csv',index=False)
 pd.DataFrame(pergame).to_csv(out/'error_analysis/phase2_to_phase4_per_game_metrics.csv',index=False)
 pd.DataFrame(disagree).to_csv(out/'error_analysis/phase4_bilstm_vs_hierarchical_disagreements.csv',index=False)
 pd.DataFrame(calibration).to_csv(out/'interpretability/phase2_to_phase4_probability_calibration_bins.csv',index=False)
 for task in TASKS:
  sub=df[df.task==task];fig,ax=plt.subplots(figsize=(9,4));labels=[x.replace('hierarchical_lstm','HLSTM').replace('_','+') for x in sub.architecture];xx=np.arange(len(sub));width=.34
  ax.bar(xx-width/2,sub.macro_f1*100,width,label='Macro F1');ax.bar(xx+width/2,sub.lie_f1*100,width,label='Lie F1');ax.set_xticks(xx,labels,rotation=15,ha='right');ax.set_ylabel('F1 (%)');ax.set_title(f'Phase 5 held-out test — {task}');ax.legend();fig.tight_layout();(out/'benchmarking').mkdir(exist_ok=True);fig.savefig(out/f'benchmarking/phase5_{task}_f1.png',dpi=160);plt.close(fig)
  fig,ax=plt.subplots(figsize=(8,4));sub2=sub.copy();ax.bar(np.arange(len(sub2)),sub2.fp,label='False positive');ax.bar(np.arange(len(sub2)),sub2.tp,bottom=sub2.fp,label='True positive');ax.set_xticks(np.arange(len(sub2)),labels,rotation=15,ha='right');ax.set_title(f'Phase 5 positive predictions — {task}');ax.legend();fig.tight_layout();fig.savefig(out/f'benchmarking/phase5_{task}_predicted_positives.png',dpi=160);plt.close(fig)
 dump(out/'audit/validation.json',{'phase5_confusion_matrices_validated':valid,'phase5_expected':8,'important_notes':notes,'prediction_files_phase5_supplied':sum(1 for a in ARCHS for task in TASKS if (root/f'artifacts/phase_5/predictions/{a}/{task}/test_predictions.csv').exists()),'source':'input artifact JSON files'})
 return df,notes,historic,pergame,disagree

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('.'));ap.add_argument('--output',type=Path,default=Path('artifacts/phase_6'));args=ap.parse_args();df,notes,_,_,_=generate(args.root,args.output)
 print(f'Analyzed {len(df)} Phase 5 test metric files.');print(df[['architecture','task','macro_f1','lie_f1']].to_string(index=False));print('Notes:',*notes,sep='\n - ')
if __name__=='__main__':main()
