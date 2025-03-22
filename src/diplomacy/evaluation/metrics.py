import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,average_precision_score

def calculate_metrics(y_true,y_pred,scores=None):
    result={
        'accuracy':float(accuracy_score(y_true,y_pred)),
        'precision':float(precision_score(y_true,y_pred,zero_division=0)),
        'recall':float(recall_score(y_true,y_pred,zero_division=0)),
        'f1':float(f1_score(y_true,y_pred,zero_division=0)),
        'lie_f1':float(f1_score(y_true,y_pred,pos_label=1,zero_division=0)),
        'macro_f1':float(f1_score(y_true,y_pred,average='macro',labels=[0,1],zero_division=0)),
        'confusion_matrix':confusion_matrix(y_true,y_pred,labels=[0,1]).tolist(),
    }
    result['pr_auc']=float(average_precision_score(y_true,scores)) if scores is not None and len(set(y_true))>1 else None
    return result
