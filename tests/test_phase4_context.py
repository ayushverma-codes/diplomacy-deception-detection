import json
from pathlib import Path
import pytest
import torch
from diplomacy.data.text_dataset import Vocabulary
from diplomacy.data.phase4_context import chronological_games, concatenate_context, ConversationDataset, collate_conversations
from diplomacy.models.neural.phase4 import MessageBiLSTM, ConcatenatedContextBiLSTM, HierarchicalContextLSTM
from diplomacy.training.phase4_runner import train_step, evaluate, model_factory
from diplomacy.training.lstm_trainer import build_loss
from torch.utils.data import DataLoader

@pytest.fixture
def small_project(tmp_path):
    records=[(1,['first game one','second one','third one'],[0,1,2]),(2,['another game message'],[0])]
    for split in ['train','validation','test']:
        raw=tmp_path/'data'/f'{split}.jsonl';raw.parent.mkdir(parents=True,exist_ok=True)
        proc=tmp_path/'artifacts'/'processed'/'suspected_lie'/f'{split}.jsonl'
        proc.parent.mkdir(parents=True,exist_ok=True)
        with raw.open('w') as f, proc.open('w') as g:
            for ci,(game,msgs,ids) in enumerate(records):
                obj={'game_id':game,'messages':msgs,'absolute_message_index':ids}
                f.write(json.dumps(obj)+'\n')
                for mi,message in enumerate(msgs):
                    if game==1 and mi==1: continue
                    g.write(json.dumps({'message_id':f'{game}:{ci}:{mi}','target':int(mi==2),
                                        'message':message,'target_name':'suspected_lie'})+'\n')
    return tmp_path

def test_context_preserves_game_and_annotation_mask(small_project):
    games=chronological_games(small_project,'train','suspected_lie')
    assert len(games)==2
    assert games[1][1]['target']==-100
    assert games[2][0]['game_id']==2
    concat=concatenate_context(games,before=2,max_tokens=100)
    assert len(concat)==3
    assert concat[1]['context_ids']==['1:0:0','1:0:1','1:0:2']
    assert 'another game message' not in concat[1]['message']
    assert all(x['target']>=0 for x in concat)

def test_context_window_and_target_preserved(small_project):
    games=chronological_games(small_project,'train','suspected_lie')
    rows=concatenate_context(games,before=1,max_tokens=3)
    assert rows[1]['context_ids']==['1:0:1','1:0:2']
    assert rows[1]['message'].endswith('third one')
    with pytest.raises(ValueError):concatenate_context(games,include_target=False)

def test_hierarchical_collation_and_labels(small_project):
    games=chronological_games(small_project,'train','suspected_lie')
    vocab=Vocabulary.fit(r['message'] for g in games.values() for r in g)
    ds=ConversationDataset(games,vocab,max_tokens=12,max_messages=2)
    batch=collate_conversations([ds[0],ds[2]])
    assert batch['input_ids'].shape[:2]==(2,2)
    assert batch['targets'][0,1].item()==-100
    assert batch['message_mask'][1,1].item()==False
    assert batch['message_ids'][0]==['1:0:0','1:0:1']

def test_models_forward_and_padding_invariance():
    vocab_size=24
    for cls in [MessageBiLSTM,ConcatenatedContextBiLSTM]:
        m=cls(vocab_size,embedding_dim=8,hidden_size=5,dropout=0.,freeze_embeddings=False).eval()
        a=torch.tensor([[2,3,4]])
        b=torch.tensor([[2,3,4,0,0]])
        with torch.no_grad():
            assert torch.allclose(m(a,torch.tensor([3])),m(b,torch.tensor([3])),atol=1e-6)
    m=HierarchicalContextLSTM(vocab_size,embedding_dim=8,hidden_size=5,context_hidden_size=6,dropout=0.,freeze_embeddings=False).eval()
    x=torch.tensor([[[2,3,4],[4,5,0]]])
    y=torch.tensor([[[2,3,4,0,0],[4,5,0,0,0]]])
    lengths=torch.tensor([[3,2]])
    mask=torch.tensor([[True,True]])
    with torch.no_grad():
        assert m(x,lengths,mask).shape==(1,2,2)
        assert torch.allclose(m(x,lengths,mask),m(y,lengths,mask),atol=1e-6)

def test_hierarchical_forward_is_causal():
    torch.manual_seed(123)
    m=HierarchicalContextLSTM(20,embedding_dim=6,hidden_size=4,context_hidden_size=5,dropout=0.).eval()
    first=torch.tensor([[[2,3],[4,5]]])
    changed=torch.tensor([[[2,3],[8,9]]])
    length=torch.tensor([[2,2]])
    mask=torch.ones((1,2),dtype=torch.bool)
    with torch.no_grad():
        a=m(first,length,mask);b=m(changed,length,mask)
    assert torch.allclose(a[:,0],b[:,0],atol=1e-6)

def test_training_and_evaluation_mask(small_project):
    games=chronological_games(small_project,'train','suspected_lie')
    vocab=Vocabulary.fit(r['message'] for g in games.values() for r in g)
    dataset=ConversationDataset(games,vocab,max_tokens=12,max_messages=3)
    loader=DataLoader(dataset,batch_size=2,collate_fn=collate_conversations)
    m=HierarchicalContextLSTM(len(vocab),embedding_dim=6,hidden_size=4,context_hidden_size=5,
                               dropout=0.,freeze_embeddings=False)
    loss,_=build_loss([0,1,0],'balanced',device='cpu')
    optimizer=torch.optim.Adam(m.parameters(),lr=.01)
    batch=next(iter(loader))
    assert train_step(m,batch,loss,optimizer,torch.device('cpu'),'hierarchical_lstm') is not None
    score,pred=evaluate(m,loader,loss,torch.device('cpu'),'hierarchical_lstm')
    assert score['n_samples']==3
    assert len(pred)==3
    assert set(x['message_id'] for x in pred)=={'1:0:0','1:0:2','2:1:0'}

def test_checkpoint_and_parity_roundtrip(tmp_path):
    from diplomacy.training.phase4_runner import save_state, load_trained, predict, predict_conversation, save_parity_reference, verify_parity_reference
    import json
    for arch in ('bilstm','bilstm_concat_context','hierarchical_lstm'):
        cfg={'architecture':arch,'task':'actual_lie','data':{'max_tokens':12},
             'context':{'before':2,'after':0,'separator':' <CTX> ','max_messages':3},
             'model':{'embedding_dim':6,'hidden_size':4,'dropout':0.,'freeze_embeddings':False},
             'seed':9}
        if arch=='hierarchical_lstm':cfg['model'].update({'context_hidden_size':5,'context_num_layers':1})
        vocab=Vocabulary.fit(['good faith','no support','perhaps yes','we agree'])
        model=model_factory(arch,len(vocab),cfg)
        optim=torch.optim.Adam(model.parameters(),lr=.01)
        state={'cfg':cfg,'vocab':vocab,'embedding_info':{'mode':'random'},'task':'actual_lie','arch':arch}
        from diplomacy.training.phase4_runner import model_dir
        p=model_dir(tmp_path,arch,'actual_lie')/'best_model.pt'
        save_state(p,model,optim,1,[],0.5,1,state)
        new,restored,metadata=load_trained(tmp_path,arch,'actual_lie')
        assert restored.to_dict()==vocab.to_dict()
        if arch=='hierarchical_lstm':
            assert len(predict_conversation(tmp_path,'actual_lie',['good faith','no support']))==2
        else:
            assert predict(tmp_path,arch,'actual_lie','good faith')['prediction'] in (0,1)
        save_parity_reference(tmp_path,arch,'actual_lie')
        assert verify_parity_reference(tmp_path,arch,'actual_lie')['passed']

def test_phase3_glove_reuse_rejects_non_glove(tmp_path):
    from diplomacy.embeddings.phase4_embeddings import initialize_from_phase3
    vocab=Vocabulary.fit(['example words'])
    with pytest.raises(FileNotFoundError):
        initialize_from_phase3(vocab,tmp_path,dim=6)

def test_phase3_frozen_glove_reuse_copies_known_tokens(tmp_path):
    from diplomacy.embeddings.phase4_embeddings import initialize_from_phase3
    path = tmp_path/'artifacts/models/neural/standard_lstm/actual_lie/best_model.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    old_vocab={'<PAD>':0,'<UNK>':1,'hello':2}
    weights=torch.arange(12,dtype=torch.float).reshape(3,4)
    torch.save({'embedding_info':{'mode':'glove'},
                'model_config':{'freeze_embeddings':True},
                'vocabulary':{'token_to_id':old_vocab},
                'model_state':{'embedding.weight':weights}},path)
    new_vocab=Vocabulary.fit(['hello unseen'])
    result,info=initialize_from_phase3(new_vocab,tmp_path,dim=4)
    assert torch.equal(result[new_vocab.token_to_id['hello']],weights[2])
    assert info['new_token_count']==1
    assert info['exact_glove_reproduction'] is False

def test_flat_phase4_predictions_retain_original_indices():
    from diplomacy.data.phase4_context import Phase4MessageDataset, collate_phase4_messages
    rows=[{'game_id':1,'message_id':'1:0:0','message':'hello','target':0,'absolute_message_index':91},
          {'game_id':1,'message_id':'1:0:1','message':'lie','target':1,'absolute_message_index':92}]
    vocab=Vocabulary.fit(x['message'] for x in rows)
    ds=Phase4MessageDataset(rows,vocab)
    batch=collate_phase4_messages([ds[0],ds[1]])
    assert batch['indices']==[91,92]
