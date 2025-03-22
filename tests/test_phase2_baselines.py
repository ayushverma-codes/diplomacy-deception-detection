import numpy as np
from diplomacy.models.baselines.random_baseline import random_predict
from diplomacy.models.baselines.majority_baseline import majority_predict
from diplomacy.models.baselines.logistic_regression import make_logistic
from diplomacy.evaluation.metrics import calculate_metrics
from diplomacy.training.baseline_trainer import select_logistic
from diplomacy.training.experiment_runner import SPECS
from diplomacy.data.preprocessing import make_task_dataset


def test_random_reproducible():
    y=np.array([0,0,0,1])
    assert np.array_equal(random_predict(y,30,7)[0],random_predict(y,30,7)[0])


def test_majority_selection():
    pred,_,majority=majority_predict([0,0,1],3)
    assert majority==0 and list(pred)==[0,0,0]


def test_logistic_balanced():
    X=np.array([[0],[1],[2],[3],[4],[5]],dtype=float);y=np.array([0,0,0,0,1,1])
    model=make_logistic(C=1,class_weight='balanced').fit(X,y)
    assert len(model.predict(X))==6
    best=select_logistic(X,y,X,y,{'C':[1.0],'class_weight':[None,'balanced']})
    assert best['class_weight'] in (None,'balanced')


def test_metrics_positive_class():
    m=calculate_metrics([0,1,1,0],[0,1,0,0],[0.1,0.8,0.2,0.3])
    assert m['confusion_matrix']==[[2,0],[1,1]]
    assert m['lie_f1']>0


def test_missing_receiver_label():
    rows=[{'sender_label':False,'receiver_label':'NOANNOTATION'},
          {'sender_label':True,'receiver_label':False}]
    assert len(make_task_dataset(rows,'actual_lie'))==2
    assert len(make_task_dataset(rows,'suspected_lie'))==1
    assert make_task_dataset(rows,'suspected_lie')[0]['target']==1


def test_ten_experiments():
    assert len(SPECS)==10
    assert len(set(x[0] for x in SPECS))==10
