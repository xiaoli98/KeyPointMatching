import nltk
from nltk import word_tokenize as tokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from nltk.stem import PorterStemmer, WordNetLemmatizer 
from typing import List

from tokenizers import Tokenizer, PreTokenizedString, NormalizedString
from tokenizers.models import WordPiece
from tokenizers import pre_tokenizers

class CustomPreTokenizer:
    """See. https://github.com/huggingface/tokenizers/blob/b24a2fc/bindings/python/examples/custom_components.py"""  
      
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))      
        
    
    def custom_actions(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        
        tokens = tokenizer(str(normalized_string))
        
        words = [word for word in tokens if word.lower() not in self.stopwords]
        words = [self.stemmer.stem(word) for word in words]
        words = [self.lemmatizer.lemmatize(word) for word in words]
     
        return words
                       
    def pre_tokenize(self, pretok: PreTokenizedString):#text: List[str]) -> str:   
        return pretok.split(self.custom_actions)
        