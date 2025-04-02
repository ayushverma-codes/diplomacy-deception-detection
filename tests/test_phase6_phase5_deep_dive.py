import numpy as np
import pandas as pd
import pytest
from diplomacy.analysis.phase6_phase5_deep_dive import select_threshold, measure, probability_summary


def test_measure_counts_and_positive_class():
    m=measure(np.array([0,0,1,1]),np.array([0,1,1,0]))
    assert m['confusion_matrix']==[[1,1],[1,1]]
    assert m['lie_f1']==.5


def test_select_threshold_uses_supplied_validation_only():
    val=pd.DataFrame({'target':[0,0,1,1],'prob_lie':[.05,.18,.19,.32]})
    t,_=select_threshold(val)
    assert t>.18 and t<=.19


def test_probability_diagnostics():
    df=pd.DataFrame({'target':[0,0,1,1],'prob_lie':[.1,.2,.8,.9]})
    s=probability_summary(df)
    assert s['roc_auc']==1.0
    assert s['mean_positive']>s['mean_negative']


def test_select_threshold_does_not_mutate_dataframe():
    df=pd.DataFrame({'target':[0,1],'prob_lie':[.2,.4]})
    before=df.copy(deep=True)
    select_threshold(df)
    pd.testing.assert_frame_equal(before,df)
