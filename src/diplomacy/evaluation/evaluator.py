from .metrics import calculate_metrics

def evaluate(y_true,y_pred,scores=None):return calculate_metrics(y_true,y_pred,scores)
