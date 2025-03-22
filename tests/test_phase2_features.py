import numpy as np
import pytest
from scipy.sparse import csr_matrix
from diplomacy.features.harbingers_features import HarbingersFeatures, FEATURE_NAMES
from diplomacy.features.bow_features import make_bow
from diplomacy.features.numerical_features import NumericalFeatures
from diplomacy.features.power_features import extract_power, fit_power_imputer
from diplomacy.features.feature_combiner import combine


def test_harbingers_representative():
    h=HarbingersFeatures(); m=h.transform(['I believe that because of this, later we win',''])
    assert m.shape==(2,8)
    assert np.all(m[1]==0)
    assert m[0,list(FEATURE_NAMES).index('claim')]>0
    assert m[0,list(FEATURE_NAMES).index('contingency')]>0


def test_bow_train_only_vocabulary():
    bow=make_bow()
    bow.fit(['alpha beta','beta gamma'])
    assert 'alpha' in bow.vocabulary_
    assert 'unknown' not in bow.vocabulary_
    m=bow.transform(['unknown alpha'])
    assert m.shape[1]==3
    assert 'unknown' not in bow.vocabulary_


def test_numerical_empty_and_length():
    output=NumericalFeatures().transform(['','hello you'])
    assert output.shape==(2,4)
    assert output[0,0]==0
    assert output[1,0]==2


def test_power_missing():
    m=extract_power([{'score_delta':'-3'},{'score_delta':None},{'score_delta':'2'}])
    assert m[0,0]==-3 and np.isnan(m[1,0])
    imputer=fit_power_imputer(m)
    assert np.isfinite(imputer.transform(m)).all()


def test_sparse_combine_rows():
    joined=combine(csr_matrix(np.ones((3,2))),np.ones((3,1)))
    assert joined.shape==(3,3)
    with pytest.raises(ValueError):combine(np.ones((3,2)),np.ones((2,1)))
