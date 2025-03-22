from scipy.sparse import csr_matrix, hstack, issparse

def combine(*parts):
    lengths={p.shape[0] for p in parts}
    if len(lengths)!=1:
        raise ValueError('Row counts do not match')
    return hstack([p if issparse(p) else csr_matrix(p) for p in parts],format='csr')
