"""Bag of words, fit vocabulary ONLY on training messages."""
from sklearn.feature_extraction.text import CountVectorizer

def make_bow(**kwargs):
    return CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b', **kwargs)
