import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from diplomacy.data.text_dataset import Vocabulary, MessageDataset, tokenize, load_rows
from diplomacy.data.text_collator import collate_messages
from diplomacy.embeddings.glove_embeddings import initialize_embeddings
from diplomacy.models.neural.lstm import StandardLSTM
from diplomacy.training.lstm_trainer import (seed_everything, build_loss, evaluate,
    metrics, save_checkpoint, load_checkpoint, train)
from diplomacy.inference.standard_lstm import load_predictor

@pytest.fixture
def records():
    return [{'message_id':'1:0:0','game_id':1,'target':0,'message':'I will help you'},
            {'message_id':'1:0:1','game_id':1,'target':1,'message':'no never!'},
            {'message_id':'1:0:2','game_id':1,'target':0,'message':'   '}]

def config():
    return {'task':'actual_lie','seed':1994, 'data':{'max_tokens':32},
            'model':{'embedding_dim':8,'hidden_size':4,'bidirectional':True,'num_layers':1,
                     'dropout':0.,'freeze_embeddings':False},
            'training':{'batch_size':2,'learning_rate':.01,'epochs':2,'patience':5,
                        'gradient_clip':1.,'class_weight_mode':'balanced','positive_weight':30.}}

def test_tokenization_and_empty():
    assert tokenize('HELLO, world!') == ['hello', ',', 'world', '!']
    assert tokenize('  ') == ['<UNK>']

def test_vocab_train_only_and_unknown():
    vocab=Vocabulary.fit(['Hello, World'], lowercase=True)
    assert vocab.token_to_id['<PAD>']==0 and vocab.token_to_id['<UNK>']==1
    assert vocab.encode('unknownword')[0] == 1
    assert 'validationword' not in vocab.token_to_id
    assert Vocabulary.from_dict(vocab.to_dict()).encode('hello')==vocab.encode('hello')

def test_embeddings():
    v=Vocabulary.fit(['hello world'])
    w, info=initialize_embeddings(v, dim=8, mode='random', seed=1994)
    assert w.shape==(len(v), 8) and torch.all(w[0]==0)
    assert info['mode']=='random'

def test_collate_and_empty(records):
    v=Vocabulary.fit(r['message'] for r in records)
    b=collate_messages([MessageDataset(records,v)[i] for i in range(3)])
    assert b['input_ids'].shape[0]==3
    assert b['lengths'].tolist()==[4,3,1]
    assert b['message_ids']==['1:0:0','1:0:1','1:0:2']

def test_model_forward_and_padding_invariance(records):
    seed_everything(1994)
    vocab=Vocabulary.fit(r['message'] for r in records)
    model=StandardLSTM(len(vocab),embedding_dim=8,hidden_size=4,dropout=0.,freeze_embeddings=False)
    model.eval()
    data=MessageDataset(records,vocab)
    short=collate_messages([data[0]])
    long=collate_messages([data[0],data[1]])
    extra=torch.cat([long['input_ids'],torch.zeros(2,5,dtype=torch.long)],dim=1)
    with torch.no_grad():
        a=model(short['input_ids'],short['lengths'])
        b=model(long['input_ids'],long['lengths'])
        c=model(extra,long['lengths'])
    assert a.shape==(1,2)
    assert torch.allclose(a,b[:1],atol=1e-6)
    assert torch.allclose(b,c,atol=1e-6)

def test_labels_and_missing_receiver(tmp_path):
    f=tmp_path/'data.jsonl'
    f.write_text(json.dumps({'target':1,'target_name':'suspected_lie','message':'hi','message_id':'a','game_id':2})+'\n')
    assert load_rows(f,'suspected_lie')[0]['target']==1
    with pytest.raises(ValueError): load_rows(f,'actual_lie')
    f.write_text(json.dumps({'target':None,'target_name':'suspected_lie','message':'hi','message_id':'a','game_id':2})+'\n')
    with pytest.raises(ValueError): load_rows(f,'suspected_lie')

def test_weighted_loss():
    loss, w=build_loss([0,0,0,1],weighting='balanced')
    assert w[1]>w[0]
    assert torch.isfinite(loss(torch.zeros(2,2),torch.tensor([0,1])))

def test_metrics():
    score=metrics([0,1],[0,1],[.1,.9])
    assert score['macro_f1']==1 and score['lie_f1']==1

def test_training_checkpoint_resume_and_cpu_inference(records,tmp_path):
    seed_everything(1994)
    cfg=config()
    v=Vocabulary.fit(r['message'] for r in records)
    loader=DataLoader(MessageDataset(records,v),batch_size=2,collate_fn=collate_messages)
    model=StandardLSTM(len(v),**cfg['model'])
    folder=tmp_path/'actual_lie'
    out=train(model, loader,loader,[r['target'] for r in records],cfg,v,
              {'mode':'random'},'actual_lie',folder,torch.device('cpu'),resume=False,epochs_override=1)
    assert (folder/'best_model.pt').is_file()
    assert (folder/'latest_checkpoint.pt').is_file()
    result=load_predictor(folder/'best_model.pt')
    p=result.predict('I will help you')
    assert p['prediction'] in (0,1) and 0<=p['prob_lie']<=1
    result2=load_predictor(folder/'best_model.pt')
    assert result2.predict('I will help you')['prob_lie']==p['prob_lie']
    state=load_checkpoint(folder/'latest_checkpoint.pt')
    assert state['epoch']==1 and state['task']=='actual_lie'
    model2=StandardLSTM(len(v),**cfg['model'])
    out2=train(model2,loader,loader,[r['target'] for r in records],cfg,v,
               {'mode':'random'},'actual_lie',folder,torch.device('cpu'),resume=True,epochs_override=2)
    assert load_checkpoint(folder/'latest_checkpoint.pt')['epoch']==2

def test_parity_reference_cpu(records, tmp_path):
    from diplomacy.training.standard_lstm_runner import save_parity_reference, verify_parity_reference
    seed_everything(1994)
    cfg=config()
    vocab=Vocabulary.fit([r['message'] for r in records])
    model=StandardLSTM(len(vocab),**cfg['model'])
    optimizer=torch.optim.Adam(model.parameters(), lr=.01)
    folder=tmp_path/'artifacts'/'models'/'neural'/'standard_lstm'/'actual_lie'
    save_checkpoint(folder/'best_model.pt',model,optimizer,1,[],.51,1,
                    cfg['model'],vocab,cfg,{'mode':'random'},'actual_lie',1994)
    path=save_parity_reference(tmp_path,'actual_lie',device='cpu')
    assert path.exists()
    result=verify_parity_reference(tmp_path,'actual_lie')
    assert result['passed'] and result['checked']==3
