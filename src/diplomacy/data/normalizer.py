"""Conservative non-learned text normalization."""
import unicodedata

def normalize_text(value: str) -> str:
    """Normalize Unicode NFC/newline conventions; preserve case, whitespace and punctuation."""
    return unicodedata.normalize('NFC', value.replace('\r\n', '\n').replace('\r', '\n'))
