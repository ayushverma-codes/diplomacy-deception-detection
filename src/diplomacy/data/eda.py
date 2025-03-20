"""Shared, model-free exploratory data analysis utilities."""
from pathlib import Path
from collections import Counter
import pandas as pd
from .loader import SPLITS,load_split
from .conversation_builder import build_messages

def load_eda_frame(root:Path)->pd.DataFrame:
    rows=[]
    for split in SPLITS:
        rows.extend(build_messages(load_split(root/'data',split),split))
    frame=pd.DataFrame(rows)
    frame['actual_lie']=frame['sender_label'].map(lambda x: x is False)
    frame['suspected_lie']=frame['receiver_label'].map(lambda x: x is False if type(x) is bool else None)
    frame['receiver_annotated']=frame['receiver_label'].map(lambda x: type(x) is bool)
    frame['word_count']=frame['message'].map(lambda x:len(x.split()) if isinstance(x,str) else 0)
    frame['char_count']=frame['message'].map(lambda x:len(x) if isinstance(x,str) else 0)
    frame['score_delta_numeric']=pd.to_numeric(frame['score_delta'],errors='coerce')
    return frame

def word_counts(texts, top_n=20):
    import re
    counter=Counter()
    for text in texts.dropna():counter.update(re.findall(r"\b[a-z]+(?:'[a-z]+)?\b",str(text).lower()))
    return pd.DataFrame(counter.most_common(top_n),columns=['token','frequency'])
