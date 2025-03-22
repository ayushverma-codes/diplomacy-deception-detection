"""ACL 2020 Table 8 Harbingers lists. WARNING: subjectivity list is truncated in paper.

All entries below are transcribed from the paper's printed lists. The subjectivity
feature covers ONLY the published beginning of that list and is not an exact
reproduction of the authors' full subjectivity dictionary.
"""
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

LEXICON = {
'claim': "accordingly|as a result|consequently|conclude that|clearly|demonstrates that|entails|follows that|hence|however|implies|in fact|in my opinion|in short|in conclusion|indicates that|it follows that|it is highly probable that|it is my contention|it should be clear that|i believe|i mean|i think|must be that|on the contrary|points to the conclusions|proves that|shows that|so|suggests that|the most obvious explanation|the point i'm trying to make|therefore|thus|the truth of the matter|to sum up|we may deduce",
'subjectivity': "abandoned|abandonment|abandon|abase|abasement|abash|abate|abdicate|aberration|abhor|abhorred|abhorrence|abhorrent|abhorrently|abhors|abidance|abide|abject|abjectly|abjure|abilities|ability|able|abnormal|abolish|abominable|abominably|abominate|abomination|above|above-average|abound|abrade|abrasive|abrupt|abscond|absence|absentee|absent-minded|absolve|absolute|absolutely|absorbed|absurd|absurdity|absurdly|absurdness|abundant|abundance|abuse|abuses|abusive|abysmal|abysmally|abyss|accede|accentuate|accept|acceptance|acceptable|accessible|accidental|acclaim|acclaimed|acclamation|accolade|accolades|accommodative|accomplish|accomplishment|accomplishments|accord|accordance|accordantly|accost|accountable|accurate|accurately|accursed|accusation|accusations|accuse|accuses|accusing|accusingly|acerbate|acerbic|acerbically|ache|achievable|achieve|achievement|achievements|acknowledge|acknowledgement|acquit|acrid|acridly|acridness|acrimonious|acrimoniously|acrimony|active|activist|actual|actuality|actually|acumen|adamant|adamantly|adaptable|adaptability|adaptive|addict|addiction|adept|adeptly|adequate|adherence|adherent|adhesion|admirable|admirer|admirably|admiration|admire|admiring|admiringly|admission|admit|admittedly|admonish|admonisher|admonishingly|admonishment|admonition",
'expansion': "additionally|also|alternatively|although|as an alternative|as if|as though|as well|besides|either or|else|except|finally|for example|for instance|further|furthermore|however|in addition|in fact|in other words|in particular|in short|in sum|in the end|in turn|indeed|instead|later|lest|likewise|meantime|meanwhile|moreover|much as|neither nor|next|nonetheless|nor|on the other hand|otherwise|overall|plus|rather|separately|similarly|specifically|then|ultimately|unless|until|when|while|yet",
'contingency': "accordingly|as a result|as long as|because|consequently|hence|if and when|if then|in the end|in turn|indeed|insofar as|lest|now that|once|since|so that|then|thereby|therefore|thus|unless|until|when",
'premise': "after all|assuming that|as|as indicated by|as shown|besides|because|deduced|derived from|due to|firstly|follows from|for|for example|for instance|for one thing|for the reason that|furthermore|given that|in addition|in light of|in that|in view of|in view of the fact that|indicated by|is supported by|may be inferred|moreover|owing to|researchers found that|secondly|this can be seen from since|since the evidence is|what's more|whereas",
'temporal_future': "after|afterward|as soon as|by then|finally|in the end|later|next|once|then|thereafter|till|ultimately|until",
'temporal_other': "also|as long as|before|before and after|earlier|in turn|meantime|meanwhile|now that|previously|simultaneously|since|still|when|when and if|while",
'comparisons': "after|although|as if|as though|besides|by comparison|by contrast|conversely|earlier|however|in contrast|in fact|in the end|indeed|instead|meanwhile|much as|neither nor|nevertheless|nonetheless|nor|on the contrary|on the one hand on the other hand|on the other hand|previously|rather|regardless|still|then|though|when|whereas|while|yet",
}
# Match complete tokens/phrases only; count non-overlapping occurrences per category.
FEATURE_NAMES = tuple(LEXICON)
def compile_category(terms):
    terms = [x.strip().replace('’', "'") for x in terms.split('|') if x.strip()]
    return re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True)) + r')(?!\w)', re.I)
PATTERNS = {name: compile_category(words) for name, words in LEXICON.items()}
class HarbingersFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.asarray([[len(p.findall(str(text or '').lower().replace('’', "'"))) for p in PATTERNS.values()] for text in X], dtype=float)
    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES)
