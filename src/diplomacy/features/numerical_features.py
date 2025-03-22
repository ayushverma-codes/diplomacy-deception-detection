"""Prediction-time message-level numeric length statistics."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
class NumericalFeatures(BaseEstimator, TransformerMixin):
    NAMES=('words','chars','avg_word_length','unique_word_ratio')
    def fit(self, X, y=None): return self
    def transform(self, X):
        rows=[]
        for text in X:
            text=str(text or '')
            words=text.split(); n=len(words)
            rows.append([n,len(text),sum(map(len,words))/n if n else 0,len(set(w.lower() for w in words))/n if n else 0])
        return np.asarray(rows,dtype=float)
    def get_feature_names_out(self,input_features=None): return np.asarray(self.NAMES)
