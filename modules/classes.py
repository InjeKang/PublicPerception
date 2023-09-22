from nltk import word_tokenize
from nltk.stem import PorterStemmer

class LemmaTokenizer:
    def __init__(self):
        self.wnl = PorterStemmer()
    def __call__(self, doc):
        return [self.wnl.stem(t) for t in word_tokenize(doc)]