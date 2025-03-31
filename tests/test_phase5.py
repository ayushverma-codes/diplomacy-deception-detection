import json
from pathlib import Path
import pytest
import torch
from torch import nn
from types import SimpleNamespace
from diplomacy.data.phase5_data import (binary_label,parse_score_delta,load_games,
      PowerScaler,Phase5ConversationDataset,collate_phase5)
from diplomacy.data.text_dataset import Vocabulary
from diplomacy.models.neural.phase5 import HierarchicalPowerLSTM,HierarchicalBERTLSTM
from diplomacy.data.phase4_context import target_preserving_text,concatenate_context
from diplomacy.training.phase5_runner import train_step,evaluate,forward,read_config
from diplomacy.training.lstm_trainer import build_loss
from torch.utils.data import DataLoader

class ToyBert(nn.Module):
    def __init__(self,dim=8):
        super().__init__();self.config=SimpleNamespace(hidden_size=dim)
        self.emb=nn.Embedding(40,dim)
    def forward(self,input_ids,attention_mask):
        words=self.emb(input_ids)*attention_mask.unsqueeze(-1)
        pooled=words.sum(1)/attention_mask.sum(1,keepdim=True).clamp(min=1)
        return SimpleNamespace(pooler_output=pooled,last_hidden_state=words)

def make_games():
    return {1:[{'game_id':1,'message_id':f'1:{j}', 'absolute_message_index':j,
             'message':t,'target':lab,'score_delta':delta} for j,(t,lab,delta) in
            enumerate([('hello there',0,2.),('maybe later',-100,None),('good support',1,-3.)])],
            2:[{'game_id':2,'message_id':'2:0','absolute_message_index':0,
                'message':'yes', 'target':0,'score_delta':0.}]}

def test_labels_power_direction_and_missing():
    assert binary_label(True)==0 and binary_label(False)==1
    assert binary_label('NOANNOTATION')==-100
    assert parse_score_delta('-3')==-3. and parse_score_delta('0')==0
    assert parse_score_delta(None) is None
    with pytest.raises(ValueError):parse_score_delta('21')
    with pytest.raises(ValueError):binary_label('unknown')

def test_original_jsonl_alignment_and_game_boundaries(tmp_path):
    rec={'game_id':1,'messages':['hello','lie'], 'sender_labels':[True,False],
       'receiver_labels':['NOANNOTATION',False],'speakers':['italy','germany'],
       'receivers':['germany','italy'],'absolute_message_index':[5,2],
       'game_score':['6','3'],'game_score_delta':['3','-3']}
    path=tmp_path/'data/train.jsonl';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rec)+'\n')
    games=load_games(tmp_path,'train','suspected_lie')
    assert [r['absolute_message_index'] for r in games[1]]==[2,5]
    assert games[1][0]['score_delta']==-3
    assert games[1][1]['target']==-100
    rec['game_score_delta']=['3']
    path.write_text(json.dumps(rec)+'\n')
    with pytest.raises(ValueError):load_games(tmp_path,'train','actual_lie')

def test_scaler_training_only_and_persistence():
    games=make_games();s=PowerScaler.fit(games)
    assert s.transform(None)==pytest.approx((s.median-s.mean)/s.std)
    assert s.transform(2)==pytest.approx(PowerScaler.from_dict(s.to_dict()).transform(2))
    assert s.transform(-3)<s.transform(2)
    with pytest.raises(ValueError):PowerScaler().transform(1)

def test_token_preservation_in_all_context_policies():
    msg='very important target'
    for prior,future in [(['a b c d'],[]),([],['after one two']),(['a b c'],['later later later']),([],[])]:
        r=target_preserving_text(prior,msg,future,max_tokens=8)
        assert 'very important target' in r
        assert len(r.split())<=8
    assert target_preserving_text(['prefix'],msg,max_tokens=2)=='very important'
    records=make_games();combined=concatenate_context(records,before=1,after=1,max_tokens=8)
    assert len(combined)==3
    assert 'good support' in combined[1]['message']

def test_collator_annotation_alignment_padding_and_games():
    games=make_games();scaler=PowerScaler.fit(games)
    vocab=Vocabulary.fit(r['message'] for group in games.values() for r in group)
    ds=Phase5ConversationDataset(games,vocab=vocab,max_messages=2,power_scaler=scaler)
    b=collate_phase5([ds[0],ds[-1]])
    assert b['input_ids'].shape[:2]==(2,2)
    assert b['targets'][0,1].item()==-100
    assert b['message_mask'][-1,1].item()==False
    assert b['game_ids']==[1,2]
    assert b['power'][0,0,0].item()==pytest.approx(scaler.transform(2),abs=1e-6)

def test_model_shapes_causality_and_power_alignment():
    m=HierarchicalPowerLSTM(40,embedding_dim=8,hidden_size=5,context_hidden_size=6,dropout=0.).eval()
    tok=torch.tensor([[[2,3,4],[5,6,0]]]);lens=torch.tensor([[3,2]])
    mask=torch.tensor([[True,True]]);power=torch.zeros(1,2,1)
    with torch.no_grad():
        a=m(tok,lens,mask,power)
        b=m(torch.tensor([[[2,3,4],[7,8,0]]]),lens,mask,power)
    assert a.shape==(1,2,2)
    assert torch.allclose(a[:,0],b[:,0],atol=1e-6)
    with pytest.raises(ValueError):m(tok,lens,mask,power[:,:1])

def test_bert_forward_freezing_padding_and_masks():
    model=HierarchicalBERTLSTM(ToyBert(),context_hidden_size=5,dropout=0,
                               use_power=True,freeze_bert=True).eval()
    assert all(not p.requires_grad for p in model.bert.parameters())
    ids=torch.tensor([[[2,3,0],[4,5,6]]]);attention=torch.tensor([[[1,1,0],[1,1,1]]]);
    mask=torch.tensor([[True,True]]);power=torch.zeros((1,2,1))
    with torch.no_grad():
        scores=model(ids,attention,mask,power)
        other=model(torch.tensor([[[2,3,0],[7,8,9]]]),attention,mask,power)
    assert scores.shape==(1,2,2)
    assert torch.allclose(scores[:,0],other[:,0],atol=1e-6)
    assert all(p.requires_grad for p in HierarchicalBERTLSTM(ToyBert(),freeze_bert=False).bert.parameters())

def test_training_and_evaluation_with_missing_annotations():
    games=make_games();scaler=PowerScaler.fit(games)
    vocab=Vocabulary.fit(r['message'] for group in games.values() for r in group)
    ds=Phase5ConversationDataset(games,vocab=vocab,max_messages=3,power_scaler=scaler)
    loader=DataLoader(ds,batch_size=2,collate_fn=collate_phase5)
    model=HierarchicalPowerLSTM(len(vocab),embedding_dim=8,hidden_size=5,context_hidden_size=5,dropout=0.,freeze_embeddings=False)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    criterion,_=build_loss([0,1,0],'balanced',device='cpu')
    batch=next(iter(loader))
    assert train_step(model,batch,'hierarchical_lstm_power',torch.device('cpu'),criterion,optimizer) is not None
    score,rows=evaluate(model,loader,'hierarchical_lstm_power',torch.device('cpu'),criterion)
    assert score['n_samples']==3
    assert len(rows)==3
    assert all(r['message_id']!='1:1' for r in rows)

@pytest.mark.parametrize('arch',["hierarchical_lstm","hierarchical_lstm_power","hierarchical_lstm_bert","hierarchical_lstm_bert_power"])
@pytest.mark.parametrize('task',["actual_lie","suspected_lie"])
def test_eight_experiment_configs_exist(arch,task):
    root=Path(__file__).resolve().parents[1]
    c=read_config(root,arch,task)
    assert c['task']==task
    assert c['architecture']==arch
    assert c['power'] is not None if arch.endswith('_power') else c['power'] is None

def test_bert_variants_training_step_with_synthetic_backbone():
    """No pretrained download or GPU required to test BERT training interface."""
    for arch in ('hierarchical_lstm_bert','hierarchical_lstm_bert_power'):
        model=HierarchicalBERTLSTM(ToyBert(),context_hidden_size=5,dropout=0,
                 use_power=arch.endswith('_power'),freeze_bert=False)
        batch={'input_ids':torch.tensor([[[2,3,0],[4,5,6]]]),
               'attention_mask':torch.tensor([[[1,1,0],[1,1,1]]]),
               'message_mask':torch.tensor([[True,True]]),
               'targets':torch.tensor([[0,1]]),'power':torch.tensor([[[0.],[1.]]])}
        criterion,_=build_loss([0,1],'balanced',device='cpu')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        original=model.classifier.weight.detach().clone()
        loss=train_step(model,batch,arch,torch.device('cpu'),criterion,optimizer)
        assert loss>0
        assert not torch.equal(original,model.classifier.weight.detach())
