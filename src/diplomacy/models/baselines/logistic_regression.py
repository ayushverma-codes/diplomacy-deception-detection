from sklearn.linear_model import LogisticRegression

def make_logistic(C=1.0,class_weight=None,seed=42):
    return LogisticRegression(C=C,class_weight=class_weight,max_iter=1500,random_state=seed,solver='liblinear')
