import pandas as pd

def compare(rows):
    return pd.DataFrame(rows).sort_values('macro_f1',ascending=False)
