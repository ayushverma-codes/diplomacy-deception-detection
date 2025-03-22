from diplomacy.models.baselines.logistic_regression import make_logistic
from diplomacy.evaluation.metrics import calculate_metrics

def select_logistic(X_train,y_train,X_val,y_val,grid,seed=42):
    """Fit on training only; optimize validation macro F1, then positive-class F1."""
    best=None
    history=[]
    for C in grid['C']:
        for weight in grid['class_weight']:
            model=make_logistic(C=C,class_weight=weight,seed=seed)
            model.fit(X_train,y_train)
            metrics=calculate_metrics(y_val,model.predict(X_val),model.predict_proba(X_val)[:,1])
            history.append({'C':C,'class_weight':weight,'macro_f1':metrics['macro_f1'],'lie_f1':metrics['lie_f1']})
            rank=(metrics['macro_f1'],metrics['lie_f1'])
            if best is None or rank>best['rank']:
                best={'model':model,'C':C,'class_weight':weight,'rank':rank,'validation':metrics}
    best['history']=history
    return best
