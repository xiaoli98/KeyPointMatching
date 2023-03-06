from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tokenizers import decoders


class KPMTokernizer():
    def __init__(self, pretrained = None, *args, **kwargs):
        """define the tokenizer pipeline, which consist of:
        1- token normalizers, we apply these normalizers as a pipe to all tokens
        2- pre tokenization process, or better how do we split the text
        3- a tokenizer algorithm, eventually specifying the unknown token 
        4- post tokenization process, how do we want the final output of our tokenizer
        """
        
        unk_token = None
        tokenizer = None
        pretrained = None
        normalizers_list = None
        pre_tokenizer_list = None
        post_processor = None
        decoder = None
        
        for kw, value in kwargs.items():
            if kw == "tokenizer":
                tokenizer = value
            elif kw == "normalizers":
                normalizers_list = value
            elif kw == "pre_tokenizer":
                pre_tokenizer_list = value
            elif kw == "post_processor":
                post_processor = value
            elif kw == "unk_token":
                unk_token = value
            elif kw == "decoder":
                decoder = value
            
        if unk_token is None:
            unk_token = "[UNK]"
        if tokenizer is None and pretrained is None:
            tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        elif pretrained is not None:
            tokenizer = Tokenizer.from_pretrained(pretrained)
        if normalizers_list is None:
            normalizers_list = [NFD(), Lowercase(), StripAccents()]
        if pre_tokenizer_list is None:
            pre_tokenizer_list = [Whitespace(), Punctuation(behavior="removed")]
        if post_processor is None:
            post_processor = TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", 1),
                    ("[SEP]", 2),
                ],
            )
        if decoder is None:
            decoder = decoders.WordPiece()
        
        self.tokenizer = tokenizer
        self.tokenizer.normalizer = normalizers.Sequence(normalizers_list)
        
        pre_tokenizer_list.append(pre_tokenizers.PreTokenizer.custom(CustomPreTokenizer))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pre_tokenizer_list)
        
        self.tokenizer.post_processor = post_processor
        self.tokenizer.decoder = decoder
        
    def train(self, files, save_path=None, trainer=WordPieceTrainer, **kwargs):
        """train the tokenizer with a trainer

        Args:
            files (string): the path of the file of text to be trained on
            save_path (string): the path to save the .json file of the train
            trainer (Trainer, optional): trainer to be used to do the tokenization. Defaults to WordPieceTrainer.
        """
        print("training the tokenizer...", end="")
        self.tokenizer.train(files, trainer(**kwargs))
        
        if save_path is not None:
            self.tokenizer.save(save_path)
        print("done!")
        
    def encode(self, text, text2=None):
        return self.tokenizer.encode(text, text2)

    def decode(self, ids):
        return self.tokenizer.decode(ids)


import nltk
from nltk import word_tokenize as tokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer as stemmer
from typing import List
#https://github.com/huggingface/tokenizers/blob/b24a2fc1781d5da4e6ebcd3ecb5b91edffc0a05f/bindings/python/examples/custom_components.py

class CustomPreTokenizer:
    
    """
    def __init__(self, *args, **kwargs):
        stop_words = None
        stem = None
        lemma = None
        
        for kw, value in kwargs.items():
            if kw == "stopwords" and value == True:
                try:
                    stop_words = stopwords.words('english')
                except LookupError:
                    nltk.download('stopwords')
            
            if kw == "stemmer" and value == True:
                stem = stemmer()
                
            if kw == "stemmer" and value == True:
                lemma = WordNetLemmatizer()
    """ 
     
    def pre_tokenize(self, text: str) -> List[str]:   
        words = perform_extra_steps(self, text)     
        return words
    
            
                
    def perform_extra_steps(self, text: str) -> List[str]:
        
        words = text.split()
        
        #if stop_words != None:
        words = remove_stopwords(self, words)
        
        #if stem != None:
        words = stemming(self, words)
            
        #if lemma != None:
        words = lemmatize(self, words)
             
        return words      
    
    def remove_stopwords(self, words):

        words = [word for word in words if word.lower() not in self.stopwords]
        # Return the list of words
        return words    
    
    def stemming(self, words):
        
        words = [self.stemmer.stem(word) for word in words]      
        return words
    
    def lemmatize(self, words):
        
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return Words
    