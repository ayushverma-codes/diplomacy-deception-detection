"""Power differential as provided by game_score_delta in original JSONL."""
import numpy as np
from sklearn.impute import SimpleImputer

def extract_power(rows):
    values=[]
    for row in rows:
        raw=row.get('score_delta')
        try: number=float(raw)
        except (TypeError,ValueError): number=np.nan
        values.append([number])
    return np.asarray(values,dtype=float)

def fit_power_imputer(training_power):
    imp=SimpleImputer(strategy='median', keep_empty_features=True)
    return imp.fit(training_power)
