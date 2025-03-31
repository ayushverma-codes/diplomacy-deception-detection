"""Reusable Phase 5 training/evaluation/CPU inference. Colab imports THIS module.

Original source: Peskov et al ACL 2020; adaptation is documented in docs.
"""
from __future__ import annotations
import argparse,csv,json,os,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score,recall_score,classification_report
from diplomacy.data.phase5_data import load_games,PowerScaler,Phase5ConversationDataset,collate_phase5
from diplomacy.data.text_dataset import Vocabulary
from diplomacy.models.neural.phase4 import HierarchicalContextLSTM
from diplomacy.models.neural.phase5 import HierarchicalPowerLSTM,HierarchicalBERTLSTM
from diplomacy.training.lstm_trainer import build_loss,seed_everything,metrics,load_checkpoint,rng_state,restore_rng
from diplomacy.training.phase4_runner import load_trained as phase4_load

ARCHITECTURES=('hierarchical_lstm','hierarchical_lstm_power','hierarchical_lstm_bert','hierarchical_lstm_bert_power')
TASKS=('actual_lie','suspected_lie')

def config_file(root,arch,task):
    if arch not in ARCHITECTURES or task not in TASKS: raise ValueError('Unknown experiment')
    return Path(root)/'configs'/'experiments'/'neural'/'phase_5'/arch/f'{task}.yaml'

def read_config(root,arch,task):
    import yaml
    cfg=yaml.safe_load(config_file(root,arch,task).read_text())
    if cfg['architecture']!=arch or cfg['task']!=task: raise ValueError('Config identity mismatch')
    return cfg

def outdir(root,arch,task):
    return Path(root)/'artifacts'/'phase_5'/'checkpoints'/arch/task

def is_bert(arch): return '_bert' in arch

def has_power(arch): return arch.endswith('_power')

def build_components(root,arch,task,device='cpu',checkpoint=None):
    root=Path(root)
    config=read_config(root,arch,task) if checkpoint is None else checkpoint['config']
    seed_everything(config['seed'])
    games={split:load_games(root,split,task) for split in ('train','validation','test')}
    sets=[set(x) for x in games.values()]
    if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]: raise ValueError('Game split leakage')
    scaler=PowerScaler.from_dict(checkpoint['power_scaler']) if checkpoint is not None else PowerScaler.fit(games['train'])
    vocab=None;tokenizer=None;backbone=None
    if is_bert(arch):
        try:
            from transformers import AutoModel,AutoTokenizer,AutoConfig
        except ImportError as exc:
            raise ImportError('Install transformers>=4.40 for BERT variants') from exc
        tokdir=root/'artifacts'/'phase_5'/'bert_tokenizer'/arch
        if checkpoint is not None:
            if not tokdir.exists():raise FileNotFoundError(f'Copy saved tokenizer directory: {tokdir}')
            tokenizer=AutoTokenizer.from_pretrained(str(tokdir),local_files_only=True)
            # No network or pretrained download needed to load trained state.
            bert_config = dict(checkpoint["bert_config"])

            model_type = bert_config.pop(
                "model_type",
                config["bert"]["model_type"]
            )

            bertcfg = AutoConfig.for_model(
                model_type,
                **bert_config
            )
            backbone=AutoModel.from_config(bertcfg)
        else:
            tokenizer=AutoTokenizer.from_pretrained(config['bert']['checkpoint'])
            backbone=AutoModel.from_pretrained(config['bert']['checkpoint'])
            tokdir.mkdir(parents=True,exist_ok=True)
            tokenizer.save_pretrained(tokdir)
        model=HierarchicalBERTLSTM(backbone,context_hidden_size=config['model']['context_hidden_size'],
                  dropout=config['model']['dropout'],use_power=has_power(arch),
                  freeze_bert=config['bert']['freeze'])
    else:
        # Reuse the exact Phase 4 vocabulary/embedding and, optionally, its trained weights.
        old_model,old_vocab,old_state=phase4_load(root,'hierarchical_lstm',task,'cpu')
        vocab=old_vocab
        if checkpoint is not None:
            vocab=Vocabulary.from_dict(checkpoint['vocabulary'])
        model_args=dict(config['model'])
        cls=HierarchicalPowerLSTM if has_power(arch) else HierarchicalContextLSTM
        model=cls(len(vocab),**model_args)
        if checkpoint is None and config['initialization']['reuse_phase4_encoder']:
            current=model.state_dict();prior=old_model.state_dict()
            for k,v in prior.items():
                if k in current and current[k].shape==v.shape:current[k]=v
            model.load_state_dict(current)
    loaders={}
    for split in games:
        ds=Phase5ConversationDataset(games[split],vocab=vocab,tokenizer=tokenizer,
                    max_messages=config['data']['max_messages'],max_tokens=config['data']['max_tokens'],
                    power_scaler=scaler)
        loaders[split]=DataLoader(ds,batch_size=config['training']['batch_size'],
                    shuffle=split=='train',collate_fn=collate_phase5,num_workers=0)
    model.to(device)
    labels=[r['target'] for g in games['train'].values() for r in g if r['target']>=0]
    criterion,weights=build_loss(labels,config['training']['class_weight_mode'],
                            config['training']['positive_weight'],device)
    return dict(model=model,loaders=loaders,vocab=vocab,tokenizer=tokenizer,
                config=config,scaler=scaler,criterion=criterion,class_weights=weights)

def forward(model,batch,arch,device):
    ids=batch['input_ids'].to(device);mask=batch['message_mask'].to(device)
    if is_bert(arch):
        return model(ids,batch['attention_mask'].to(device),mask,
                     power=batch['power'].to(device) if has_power(arch) else None)
    if has_power(arch):
        return model(ids,batch['lengths'].to(device),mask,batch['power'].to(device))
    return model(ids,batch['lengths'].to(device),mask)

def train_step(model,batch,arch,device,criterion,optimizer,scaler=None,accumulate=1,clip=1.,update=True):
    model.train()
    amp=bool(scaler is not None and scaler.is_enabled())
    with torch.autocast(device_type='cuda',enabled=amp):
        logits=forward(model,batch,arch,device)
        labels=batch['targets'].to(device)
        valid=(labels!=-100)&batch['message_mask'].to(device)
        if not valid.any():return None
        loss=criterion(logits[valid],labels[valid])
    if amp:scaler.scale(loss/accumulate).backward()
    else:(loss/accumulate).backward()
    if update:
        if amp:scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],clip)
        if amp:scaler.step(optimizer);scaler.update()
        else:optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return float(loss.item())

@torch.no_grad()
def evaluate(model,loader,arch,device,criterion):
    model.eval();true=[];pred=[];prob=[];rows=[];total=0.;denom=0.
    for batch in loader:
        logits=forward(model,batch,arch,device)
        labels=batch['targets'].to(device)
        valid=(labels!=-100)&batch['message_mask'].to(device)
        if not valid.any():continue
        loss=nn.functional.cross_entropy(logits[valid],labels[valid],weight=criterion.weight,reduction='none')
        total+=loss.sum().item();denom+=criterion.weight[labels[valid]].sum().item()
        for b in range(labels.shape[0]):
            for m in range(len(batch['message_ids'][b])):
                if not valid[b,m]:continue
                score=float(logits[b,m].softmax(-1)[1].item())
                guess=int(score>=.5);actual=int(labels[b,m].item())
                true.append(actual);pred.append(guess);prob.append(score)
                rows.append({'game_id':batch['game_ids'][b],
                       'message_id':batch['message_ids'][b][m],
                       'absolute_message_index':batch['indices'][b][m],
                       'target':actual,'prediction':guess,'prob_lie':score,
                       'prob_truth':1-score})
    if not true:raise ValueError('Evaluation has no annotated messages')
    result=metrics(true,pred,prob)
    result.update({'loss':total/max(denom,1e-12),
                   'macro_precision':float(precision_score(true,pred,average='macro',zero_division=0)),
                   'macro_recall':float(recall_score(true,pred,average='macro',zero_division=0)),
                   'classification_report':classification_report(true,pred,labels=[0,1],output_dict=True,zero_division=0)})
    return result,rows

def write_predictions(root,arch,task,split,score,rows):
    base=Path(root)/'artifacts'/'phase_5'
    metric=base/'metrics'/arch/task;metric.mkdir(parents=True,exist_ok=True)
    (metric/f'{split}_metrics.json').write_text(json.dumps(score,indent=2))
    dest=base/'predictions'/arch/task;dest.mkdir(parents=True,exist_ok=True)
    with (dest/f'{split}_predictions.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def save_state(path,components,arch,task,optimizer,scheduler,scaler,epoch,best,best_epoch,history):
    model=components['model'];vocab=components['vocab']
    state={'architecture':arch,'task':task,'config':components['config'],
           'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
           'scheduler_state':scheduler.state_dict() if scheduler else None,
           'amp_scaler_state':scaler.state_dict() if scaler else None,
           'power_scaler':components['scaler'].to_dict(),
           'vocabulary':vocab.to_dict() if vocab else None,
           'bert_config':model.bert.config.to_dict() if is_bert(arch) else None,
           'epoch':epoch,'best_score':best,'best_epoch':best_epoch,'history':history,
           'rng_state':rng_state(),'versions':{'torch':torch.__version__}}
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(state,temp);os.replace(temp,path)

def run_training(root,arch,task,device=None,resume=True,epochs_override=None):
    root=Path(root);device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    folder=outdir(root,arch,task);folder.mkdir(parents=True,exist_ok=True)
    latest=folder/'latest_checkpoint.pt';best=folder/'best_model.pt'
    (folder/'experiment_config.json').write_text(json.dumps(read_config(root,arch,task),indent=2))
    recovered=load_checkpoint(latest,'cpu') if resume and latest.exists() else None
    p=build_components(root,arch,task,device,checkpoint=recovered)
    if recovered is not None and recovered['config']!=read_config(root,arch,task):
        raise ValueError('Resume config differs from persisted experiment')
    model=p['model'];cfg=p['config'];tcfg=cfg['training']
    optimizer=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=float(tcfg['learning_rate']))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:1.)
    amp_enabled=device.type=='cuda' and bool(tcfg['mixed_precision'])
    amp_scaler=torch.amp.GradScaler('cuda',enabled=amp_enabled)
    start=0;best_score=-1.;best_epoch=0;history=[]
    if recovered:
        model.load_state_dict(recovered['model_state'])
        optimizer.load_state_dict(recovered['optimizer_state'])
        scheduler.load_state_dict(recovered['scheduler_state'])
        if recovered.get('amp_scaler_state'):amp_scaler.load_state_dict(recovered['amp_scaler_state'])
        start=int(recovered['epoch']);best_score=float(recovered['best_score']);best_epoch=int(recovered['best_epoch']);history=list(recovered['history'])
        restore_rng(recovered.get('rng_state'))
    max_epochs=int(epochs_override or tcfg['epochs']);patience=int(tcfg['patience']);accum=int(tcfg['gradient_accumulation'])
    print(f'Phase5 {arch}/{task}: device={device} epoch_start={start} batches={len(p["loaders"]["train"])}')
    for epoch in range(start,max_epochs):
        if epoch-best_epoch>=patience:break
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for ix,batch in enumerate(p['loaders']['train']):
            update=((ix+1)%accum==0 or ix+1==len(p['loaders']['train']))
            try:loss=train_step(model,batch,arch,device,p['criterion'],optimizer,amp_scaler,accum,tcfg['gradient_clip'],update)
            except torch.cuda.OutOfMemoryError as exc:
                optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
                raise RuntimeError('CUDA OOM: choose explicit smaller batch/max_messages/max_tokens; no architecture changed') from exc
            if loss is not None:losses.append(loss)
        scheduler.step()
        validation,_=evaluate(model,p['loaders']['validation'],arch,device,p['criterion'])
        record={'epoch':epoch+1,'train_loss':sum(losses)/max(len(losses),1),
                'validation_loss':validation['loss'],'validation_macro_f1':validation['macro_f1'],
                'validation_lie_f1':validation['lie_f1']}
        history.append(record)
        logdir=root/'artifacts'/'phase_5'/'logs'/arch/task
        logdir.mkdir(parents=True,exist_ok=True)
        with (logdir/'epochs.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(record)+'\n')
        if validation['macro_f1']>best_score:
            best_score=validation['macro_f1'];best_epoch=epoch+1
            save_state(best,p,arch,task,optimizer,scheduler,amp_scaler,epoch+1,best_score,best_epoch,history)
        save_state(latest,p,arch,task,optimizer,scheduler,amp_scaler,epoch+1,best_score,best_epoch,history)
        print(f'{arch}/{task} epoch={epoch+1} val_macro_f1={validation["macro_f1"]:.4f}')
    summary={'architecture':arch,'task':task,'best_epoch':best_epoch,'best_validation_macro_f1':best_score,'history':history}
    (folder/'training_summary.json').write_text(json.dumps(summary,indent=2))
    return summary

def load_trained(root,arch,task,device='cpu'):
    checkpoint=load_checkpoint(outdir(root,arch,task)/'best_model.pt','cpu')
    if checkpoint['architecture']!=arch or checkpoint['task']!=task: raise ValueError('Checkpoint identity mismatch')
    p=build_components(root,arch,task,device,checkpoint=checkpoint)
    p['model'].load_state_dict(checkpoint['model_state']);p['model'].eval()
    p['checkpoint']=checkpoint
    return p

def run_evaluation(root,arch,task,split='test',device='cpu'):
    if split not in ('validation','test'):raise ValueError('Invalid split')
    p=load_trained(root,arch,task,device)
    score,rows=evaluate(p['model'],p['loaders'][split],arch,torch.device(device),p['criterion'])
    score.update({'architecture':arch,'task':task,'split':split,'best_epoch':p['checkpoint']['best_epoch']})
    write_predictions(root,arch,task,split,score,rows)
    return score,rows

def predict_conversation(root,arch,task,messages,power_deltas=None,device='cpu'):
    """A single conversation in chronological order; optional power per message.

    The trained checkpoint includes power imputation/normalization; the same method
    is applied here and during training. Chunks reset at max_messages.
    """
    if not messages:return []
    if power_deltas is not None and len(power_deltas)!=len(messages):raise ValueError('Power/text alignment mismatch')
    p=load_trained(root,arch,task,device)
    data=[{'game_id':'inference','message_id':f'inference:{i}','absolute_message_index':i,
           'message':m,'target':-100,'score_delta':None if power_deltas is None else power_deltas[i]}
          for i,m in enumerate(messages)]
    games={'inference':data}
    ds=Phase5ConversationDataset(games,vocab=p['vocab'],tokenizer=p['tokenizer'],
         max_messages=p['config']['data']['max_messages'],max_tokens=p['config']['data']['max_tokens'],
         power_scaler=p['scaler'])
    output=[]
    p['model'].eval()
    with torch.no_grad():
        for sample in ds:
            batch=collate_phase5([sample]);logits=forward(p['model'],batch,arch,torch.device(device))
            scores=logits.softmax(-1)[0,:len(sample['message_ids']),1].tolist()
            for i,score in enumerate(scores):
                output.append({'message_id':sample['message_ids'][i],
                     'message_position':sample['indices'][i],
                     'prediction':int(score>=.5),'prob_lie':score,'prob_truth':1-score})
    return output

def save_parity(root,arch,task):
    cases=[{'messages':['We should cooperate.','I agree, for now.'],
            'power_deltas':[0,2]}]
    for c in cases:c['predictions']=predict_conversation(root,arch,task,c['messages'],c['power_deltas'])
    path=outdir(root,arch,task)/'parity_reference.json';path.write_text(json.dumps(cases,indent=2));return path

def verify_parity(root,arch,task,tolerance=1e-4):
    path=outdir(root,arch,task)/'parity_reference.json'
    cases=json.loads(path.read_text());checked=0
    for case in cases:
        actual=predict_conversation(root,arch,task,case['messages'],case['power_deltas'])
        for got,expected in zip(actual,case['predictions']):
            assert got['prediction']==expected['prediction']
            assert abs(got['prob_lie']-expected['prob_lie'])<=tolerance
            checked+=1
    return {'architecture':arch,'task':task,'checked':checked,'passed':True}

def compare_historical(root,task):
    """Read persisted results only; Phase 5 is excluded until actually evaluated."""
    from diplomacy.training.phase4_runner import compare_previous
    df=compare_previous(root,task)
    rows=df.to_dict('records')
    for arch in ARCHITECTURES:
        path=Path(root)/'artifacts'/'phase_5'/'metrics'/arch/task/'test_metrics.json'
        if path.exists():
            result=json.loads(path.read_text())
            rows.append({'phase':'5','model':arch,**{k:result[k] for k in ('accuracy','macro_f1','lie_f1')}})
    return rows

def main():
    parser=argparse.ArgumentParser(description='Phase 5 research experiment runner')
    parser.add_argument('--root',default='.')
    parser.add_argument('--architecture',required=True,choices=ARCHITECTURES)
    parser.add_argument('--task',required=True,choices=TASKS)
    parser.add_argument('--action',choices=('train','evaluate','predict','parity','save-parity'),default='train')
    parser.add_argument('--split',choices=('validation','test'),default='validation')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--no-resume',action='store_true')
    parser.add_argument('--messages',nargs='*',default=['I promise to support you.'])
    parser.add_argument('--power-deltas',nargs='*',type=float)
    args=parser.parse_args()
    if args.action=='train':result=run_training(args.root,args.architecture,args.task,args.device,not args.no_resume,args.epochs)
    elif args.action=='evaluate':result=run_evaluation(args.root,args.architecture,args.task,args.split,args.device)[0]
    elif args.action=='predict':result=predict_conversation(args.root,args.architecture,args.task,args.messages,args.power_deltas,args.device)
    elif args.action=='save-parity':result=str(save_parity(args.root,args.architecture,args.task))
    else:result=verify_parity(args.root,args.architecture,args.task)
    print(result)
if __name__=='__main__':main()
