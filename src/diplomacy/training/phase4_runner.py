"""Single-source Phase 4 training, evaluation and local CPU inference."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from diplomacy.data.text_dataset import Vocabulary, MessageDataset, load_rows
from diplomacy.data.text_collator import collate_messages
from diplomacy.data.phase4_context import chronological_games, concatenate_context, target_preserving_text, ConversationDataset, collate_conversations, Phase4MessageDataset, collate_phase4_messages
from diplomacy.models.neural.phase4 import MessageBiLSTM, ConcatenatedContextBiLSTM, HierarchicalContextLSTM
from diplomacy.embeddings.glove_embeddings import initialize_embeddings
from diplomacy.embeddings.phase4_embeddings import initialize_from_phase3
from diplomacy.training.lstm_trainer import seed_everything, build_loss, metrics, load_checkpoint, rng_state, restore_rng

ARCHITECTURES = ('bilstm', 'bilstm_concat_context', 'hierarchical_lstm')
TASKS = ('actual_lie', 'suspected_lie')
MODELS = {'bilstm': MessageBiLSTM, 'bilstm_concat_context': ConcatenatedContextBiLSTM,
          'hierarchical_lstm': HierarchicalContextLSTM}


def config_path(root, arch, task):
    if arch not in ARCHITECTURES or task not in TASKS:
        raise ValueError('Invalid architecture/task')
    return Path(root)/'configs'/'experiments'/'neural'/'phase_4'/arch/f'{task}.yaml'


def read_config(root, arch, task):
    cfg = json.loads(config_path(root, arch, task).read_text(encoding='utf8'))
    if cfg['task'] != task or cfg['architecture'] != arch:
        raise ValueError('Configuration mismatch')
    return cfg


def model_dir(root, arch, task):
    return Path(root)/'artifacts'/'models'/'neural'/'phase_4'/arch/task


def model_factory(arch, vocab_size, cfg, embedding_weights=None):
    return MODELS[arch](vocab_size, embedding_weights=embedding_weights, **cfg['model'])


def task_data(root, arch, task, cfg):
    root = Path(root)
    games = {split: chronological_games(root, split, task) for split in ('train','validation','test')}
    game_ids = {split:set(g) for split,g in games.items()}
    assert not(game_ids['train'] & game_ids['validation'] or game_ids['train'] & game_ids['test'] or game_ids['validation'] & game_ids['test'])
    if arch == 'hierarchical_lstm':
        # All messages, including those without suspected-lie annotation, remain available as context.
        return games
    if arch == 'bilstm_concat_context':
        policy = cfg['context']
        return {split: concatenate_context(groups, before=policy['before'],after=policy['after'],
                    separator=policy['separator'],max_tokens=cfg['data']['max_tokens'],
                    lowercase=cfg['data']['lowercase'],include_target=policy['include_target'])
                for split,groups in games.items()}
    # Model 1 has no context, identical message-only inputs to Phase 3.
    return {split: load_rows(root/'artifacts'/'processed'/task/f'{split}.jsonl', task)
            for split in ('train','validation','test')}


def training_texts(rows, arch):
    if arch == 'hierarchical_lstm':
        return (r['message'] for g in rows['train'].values() for r in g)
    return (r['message'] for r in rows['train'])


def datasets_and_loaders(rows, arch, vocab, cfg, device, shuffle=True):
    loaders = {}
    for split in ('train','validation','test'):
        if arch == 'hierarchical_lstm':
            dataset = ConversationDataset(rows[split],vocab,
                                          max_tokens=cfg['data']['max_tokens'],
                                          max_messages=cfg['context']['max_messages'])
            collate = collate_conversations
        else:
            dataset = Phase4MessageDataset(rows[split],vocab,cfg['data']['max_tokens'])
            collate = collate_phase4_messages
        loaders[split] = DataLoader(dataset,batch_size=cfg['training']['batch_size'],
                                    shuffle=(split == 'train' and shuffle),collate_fn=collate,
                                    num_workers=0,pin_memory=device.type=='cuda')
    return loaders


def train_labels(rows, arch):
    if arch=='hierarchical_lstm':
        return [r['target'] for g in rows['train'].values() for r in g if r['target'] >= 0]
    return [r['target'] for r in rows['train']]


def prepare(root, arch, task, device=None, download_glove=False, resume=True):
    root=Path(root)
    cfg=read_config(root,arch,task)
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    seed_everything(cfg['seed'])
    data=task_data(root,arch,task,cfg)
    directory=model_dir(root,arch,task)
    latest=directory/'latest_checkpoint.pt'
    if resume and latest.exists():
        state=load_checkpoint(latest,'cpu')
        if state['experiment_config'] != cfg:
            raise ValueError('Cannot resume: checkpoint config differs')
        vocab=Vocabulary.from_dict(state['vocabulary'])
        weights=None
        embedding_info=state['embedding_info']
    else:
        vocab=Vocabulary.fit(training_texts(data,arch),min_freq=cfg['data']['min_freq'],
                             max_size=cfg['data']['max_vocab_size'],lowercase=cfg['data'].get('lowercase',True))
        if cfg['embeddings']['mode'] == 'phase3_glove_reuse':
            weights,embedding_info=initialize_from_phase3(vocab,root,dim=cfg['model']['embedding_dim'],seed=cfg['seed'])
        else:
            weights,embedding_info=initialize_embeddings(vocab,dim=cfg['model']['embedding_dim'],
                        mode=cfg['embeddings']['mode'],
                        glove_zip=root/cfg['embeddings']['glove_zip'],
                        download_glove=download_glove or cfg['embeddings']['download_glove'],seed=cfg['seed'])
    vocab.save(root/'artifacts'/'vocabularies'/'phase_4'/arch/task/'vocabulary.json')
    model=model_factory(arch,len(vocab),cfg,weights)
    loaders=datasets_and_loaders(data,arch,vocab,cfg,device)
    print(f'{arch}/{task}: vocab={len(vocab)}, train_samples={len(loaders["train"].dataset)}, device={device}')
    return {'root':root,'arch':arch,'task':task,'cfg':cfg,'vocab':vocab,'model':model,
            'rows':data,'loaders':loaders,'embedding_info':embedding_info,'device':device}


def forward(model, batch, device, arch):
    if arch=='hierarchical_lstm':
        return model(batch['input_ids'].to(device),batch['lengths'].to(device),batch['message_mask'].to(device))
    return model(batch['input_ids'].to(device),batch['lengths'].to(device))


def batch_targets(batch, arch, device):
    return batch['targets'].to(device)


def train_step(model,batch,criterion,optimizer,device,arch,clip=1.):
    model.train();optimizer.zero_grad(set_to_none=True)
    logits=forward(model,batch,device,arch)
    labels=batch_targets(batch,arch,device)
    if arch=='hierarchical_lstm':
        valid=labels!=-100
        if not valid.any(): return None
        logits=logits[valid];labels=labels[valid]
    loss=criterion(logits,labels)
    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),clip);optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(model,loader,criterion,device,arch):
    model.eval(); total_loss=0.;total_weight=0.;truth=[];pred=[];prob=[];output=[]
    for batch in loader:
        logits=forward(model,batch,device,arch)
        labels=batch_targets(batch,arch,device)
        if arch=='hierarchical_lstm':
            valid=(labels!=-100)&batch['message_mask'].to(device)
            flat_ids=[mid for seq in batch['message_ids'] for mid in seq]
            flat_games=[g for g,seq in zip(batch['game_ids'],batch['message_ids']) for _ in seq]
            flat_indices=[ind for seq in batch['indices'] for ind in seq]
            # flatten ids in exactly the same valid-order as flattened logits/labels
            padded_ids=[];padded_games=[];padded_indices=[]
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if valid[i,j]:
                        padded_ids.append(batch['message_ids'][i][j]);padded_games.append(batch['game_ids'][i]);padded_indices.append(batch['indices'][i][j])
            logits=logits[valid];labels=labels[valid]
            ids=padded_ids;games=padded_games;indices=padded_indices
        else:
            ids=batch['message_ids'];games=batch['game_ids'];indices=batch['indices']
        if not labels.numel(): continue
        weighted_loss=nn.functional.cross_entropy(logits,labels,weight=criterion.weight,reduction='none')
        total_loss+=weighted_loss.sum().item();total_weight+=criterion.weight[labels].sum().item()
        score=logits.softmax(dim=-1)[:,1].cpu().tolist()
        guess=logits.argmax(dim=-1).cpu().tolist()
        y=labels.cpu().tolist()
        truth.extend(y);pred.extend(guess);prob.extend(score)
        for i,mid in enumerate(ids):
            output.append({'game_id':games[i],'message_id':mid,'absolute_message_index':indices[i],
                           'target':y[i],'prediction':guess[i],'prob_lie':score[i],'prob_truth':1-score[i]})
    if not truth: raise ValueError('No annotated examples in evaluation split')
    result=metrics(truth,pred,prob);result['loss']=total_loss/max(total_weight,1e-12)
    return result,output


def save_state(path,model,optimizer,epoch,history,best_score,best_epoch,p):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state={'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
           'epoch':epoch,'history':history,'best_score':best_score,'best_epoch':best_epoch,
           'model_config':p['cfg']['model'],'vocabulary':p['vocab'].to_dict(),
           'experiment_config':p['cfg'],'embedding_info':p['embedding_info'],
           'class_mapping':{'truth':0,'lie':1},'task':p['task'],'architecture':p['arch'],
           'seed':p['cfg']['seed'],'rng_state':rng_state(),
           'versions':{'pytorch':torch.__version__}}
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)


def run_training(root,arch,task,prepared=None,resume=True,epochs_override=None,download_glove=False,device=None):
    p=prepared or prepare(root,arch,task,device=device,download_glove=download_glove,resume=resume)
    cfg=p['cfg'];device=p['device'];model=p['model'].to(device)
    criterion,weights=build_loss(train_labels(p['rows'],arch),
                 cfg['training']['class_weight_mode'],cfg['training']['positive_weight'],device)
    optimizer=torch.optim.Adam((q for q in model.parameters() if q.requires_grad),lr=float(cfg['training']['learning_rate']))
    folder=model_dir(root,arch,task);folder.mkdir(parents=True,exist_ok=True)
    latest=folder/'latest_checkpoint.pt';best=folder/'best_model.pt'
    start=0;best_score=float('-inf');best_epoch=0;history=[]
    if resume and latest.exists():
        state=load_checkpoint(latest,device)
        if state['experiment_config'] != cfg or state['vocabulary'] != p['vocab'].to_dict():
            raise ValueError('Checkpoint mismatch')
        model.load_state_dict(state['model_state']);optimizer.load_state_dict(state['optimizer_state'])
        start=state['epoch'];best_score=state['best_score'];best_epoch=state['best_epoch'];history=state['history']
        restore_rng(state.get('rng_state'))
    max_epochs=int(epochs_override or cfg['training']['epochs']);patience=cfg['training']['patience']
    started=time.time()
    for epoch in range(start,max_epochs):
        if epoch-best_epoch>=patience: break
        losses=[]
        for batch in p['loaders']['train']:
            loss=train_step(model,batch,criterion,optimizer,device,arch,cfg['training']['gradient_clip'])
            if loss is not None: losses.append(loss)
        valid,_=evaluate(model,p['loaders']['validation'],criterion,device,arch)
        row={'epoch':epoch+1,'train_loss':sum(losses)/max(len(losses),1),
             'validation_loss':valid['loss'],'validation_macro_f1':valid['macro_f1'],
             'validation_lie_f1':valid['lie_f1']}
        history.append(row)
        if valid['macro_f1']>best_score:
            best_score=valid['macro_f1'];best_epoch=epoch+1
            save_state(best,model,optimizer,epoch+1,history,best_score,best_epoch,p)
        save_state(latest,model,optimizer,epoch+1,history,best_score,best_epoch,p)
        print(f'{arch}/{task} epoch {epoch+1}: validation_macro_f1={valid["macro_f1"]:.4f}')
    summary={'architecture':arch,'task':task,'history':history,'best_epoch':best_epoch,
             'best_val_macro_f1':best_score,'class_weights':weights,
             'duration_seconds_this_session':time.time()-started,
             'embedding_info':p['embedding_info']}
    (folder/'history.json').write_text(json.dumps(history,indent=2))
    (folder/'training_summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def load_trained(root,arch,task,device='cpu'):
    path=model_dir(root,arch,task)/'best_model.pt'
    if not path.exists():raise FileNotFoundError(path)
    state=load_checkpoint(path,'cpu')
    if state['architecture']!=arch or state['task']!=task:raise ValueError('Checkpoint identity mismatch')
    vocab=Vocabulary.from_dict(state['vocabulary'])
    model=model_factory(arch,len(vocab),state['experiment_config'])
    model.load_state_dict(state['model_state'])
    model.to(device).eval()
    return model,vocab,state


def run_evaluation(root,arch,task,split='test',device='cpu'):
    if split not in ('validation','test'):raise ValueError('Invalid split')
    device=torch.device(device)
    model,vocab,state=load_trained(root,arch,task,device)
    cfg=state['experiment_config']
    data=task_data(root,arch,task,cfg)
    loaders=datasets_and_loaders(data,arch,vocab,cfg,device,shuffle=False)
    criterion,_=build_loss(train_labels(data,arch),cfg['training']['class_weight_mode'],
                            cfg['training']['positive_weight'],device)
    outcome,rows=evaluate(model,loaders[split],criterion,device,arch)
    outcome.update({'architecture':arch,'task':task,'split':split,'best_epoch':state['best_epoch']})
    folder=Path(root)/'artifacts'/'metrics'/'phase_4'/arch/task;folder.mkdir(parents=True,exist_ok=True)
    (folder/f'{split}_metrics.json').write_text(json.dumps(outcome,indent=2))
    out=Path(root)/'artifacts'/'predictions'/'phase_4'/arch/task;out.mkdir(parents=True,exist_ok=True)
    with (out/f'{split}_predictions.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return outcome,rows


def predict(root,arch,task,text,context=None,device='cpu',future_context=None):
    """Message-only or concatenated-context inference. Hierarchical requires predict_conversation."""
    if arch=='hierarchical_lstm':
        raise ValueError('Hierarchical model requires predict_conversation with ordered messages')
    model,vocab,state=load_trained(root,arch,task,device)
    cfg=state['experiment_config']
    if arch=='bilstm_concat_context':
        policy=cfg['context'];prior=list(context or [])[-policy['before']:]
        after=policy['after']
        future=list(future_context or [])
        if after and len(future)<after:
            raise ValueError('Provide following messages for a noncausal context configuration')
        text=target_preserving_text(prior,text,future[:after],
                       max_tokens=cfg['data']['max_tokens'],
                       separator=policy['separator'],lowercase=cfg['data'].get('lowercase',True))
        tokens=vocab.encode(text,cfg['data']['max_tokens'])
    else:tokens=vocab.encode(text,cfg['data']['max_tokens'])
    with torch.no_grad():
        logits=model(torch.tensor([tokens],device=device),torch.tensor([len(tokens)],device=device))
        probability=float(logits.softmax(-1)[0,1])
    return {'prediction':int(probability>=0.5),'prob_lie':probability,'prob_truth':1-probability}


def predict_conversation(root,task,messages,device='cpu'):
    """Ordered conversation / game messages; chunk boundaries match training."""
    if not messages: return []
    arch='hierarchical_lstm';model,vocab,state=load_trained(root,arch,task,device)
    cfg=state['experiment_config'];limit=cfg['context']['max_messages'];outputs=[]
    for start in range(0,len(messages),limit):
        piece=messages[start:start+limit]
        ids=[vocab.encode(m,cfg['data']['max_tokens']) for m in piece]
        tokens=torch.zeros(1,len(ids),max(map(len,ids)),dtype=torch.long,device=device)
        for i,seq in enumerate(ids):tokens[0,i,:len(seq)]=torch.tensor(seq,device=device)
        lengths=torch.tensor([[len(s) for s in ids]],device=device)
        mask=torch.ones(1,len(ids),dtype=torch.bool,device=device)
        with torch.no_grad():scores=model(tokens,lengths,mask).softmax(-1)[0,:,1].tolist()
        outputs.extend([{'message_position':start+i,'prediction':int(score>=0.5),
                         'prob_lie':score,'prob_truth':1-score} for i,score in enumerate(scores)])
    return outputs


def save_parity_reference(root,arch,task):
    samples=['I promise I will support you.', 'Perhaps tomorrow.', '']
    if arch=='hierarchical_lstm':
        cases=[{'messages':samples,'predictions':predict_conversation(root,task,samples)}]
    else:
        cases=[{'text':text,'context':(['We should cooperate.'] if arch=='bilstm_concat_context' else []),
                'prediction':predict(root,arch,task,text,['We should cooperate.'] if arch=='bilstm_concat_context' else [])}
               for text in samples]
    path=model_dir(root,arch,task)/'parity_reference.json'
    path.write_text(json.dumps(cases,indent=2));return path


def verify_parity_reference(root,arch,task,tolerance=1e-4):
    path=model_dir(root,arch,task)/'parity_reference.json'
    cases=json.loads(path.read_text())
    for case in cases:
        if arch=='hierarchical_lstm':
            current=predict_conversation(root,task,case['messages'])
            expected=case['predictions']
        else:
            current=[predict(root,arch,task,case['text'],case['context'])]
            expected=[case['prediction']]
        if len(current)!=len(expected):raise AssertionError('Prediction count mismatch')
        for a,b in zip(current,expected):
            assert a['prediction']==b['prediction']
            assert abs(a['prob_lie']-b['prob_lie'])<=tolerance
    return {'architecture':arch,'task':task,'checked':sum(len(x['predictions']) if arch=='hierarchical_lstm' else 1 for x in cases),'passed':True}


def compare_previous(root,task):
    root=Path(root);rows=[]
    import pandas as pd
    traditional=root/'artifacts'/'metrics'/'traditional'/task/'test_results.csv'
    if traditional.exists():
        df=pd.read_csv(traditional)
        for _,r in df.iterrows():rows.append({'phase':'2','model':r['model_id'],'accuracy':r['accuracy'],'macro_f1':r['macro_f1'],'lie_f1':r['lie_f1']})
    phase3=root/'artifacts'/'metrics'/'standard_lstm'/task/'test_metrics.json'
    if phase3.exists():
        r=json.loads(phase3.read_text());rows.append({'phase':'3','model':'standard_lstm',**{k:r[k] for k in ('accuracy','macro_f1','lie_f1')}})
    for arch in ARCHITECTURES:
        p=root/'artifacts'/'metrics'/'phase_4'/arch/task/'test_metrics.json'
        if p.exists():
            r=json.loads(p.read_text());rows.append({'phase':'4','model':arch,**{k:r[k] for k in ('accuracy','macro_f1','lie_f1')}})
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='.')
    parser.add_argument('--architecture',choices=ARCHITECTURES,required=True)
    parser.add_argument('--task',choices=TASKS,required=True)
    parser.add_argument('--action',choices=('train','evaluate','predict','parity'),default='train')
    parser.add_argument('--split',choices=('validation','test'),default='test')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--download-glove',action='store_true')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--text',default='I will support you next season.')
    parser.add_argument('--context',nargs='*',default=[])
    parser.add_argument('--future-context',nargs='*',default=[])
    parser.add_argument('--no-resume',action='store_true')
    args=parser.parse_args()
    if args.action=='train':print(run_training(args.root,args.architecture,args.task,resume=not args.no_resume,epochs_override=args.epochs,download_glove=args.download_glove,device=args.device))
    elif args.action=='evaluate':print(run_evaluation(args.root,args.architecture,args.task,args.split,args.device)[0])
    elif args.action=='parity':print(verify_parity_reference(args.root,args.architecture,args.task))
    elif args.architecture=='hierarchical_lstm':print(predict_conversation(args.root,args.task,args.context+[args.text],args.device))
    else:print(predict(args.root,args.architecture,args.task,args.text,args.context,args.device,args.future_context))

if __name__=='__main__':main()
