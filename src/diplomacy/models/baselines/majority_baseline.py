import numpy as np

def majority_predict(training_labels,n):
    counts=np.bincount(np.asarray(training_labels,dtype=int),minlength=2)
    label=int(np.argmax(counts))
    return np.full(n,label,dtype=int), np.full(n,label,dtype=float),label
