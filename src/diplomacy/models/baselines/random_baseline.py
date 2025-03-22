import numpy as np

def random_predict(training_labels,n,seed=42):
    probability=np.mean(training_labels)
    rng=np.random.default_rng(seed)
    return (rng.random(n)<probability).astype(int), np.full(n,probability)
