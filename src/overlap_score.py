import nltk
from nltk import StemmerI
from nltk.corpus import stopwords, wordnet
from nltk.downloader import Downloader
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import string

class overlap_score():

    def __init__(
            self,
            stemming: bool = False,
            stop_words: bool = False,
            lemmitizer: bool = False,
            language = "english"
    ):
        self.use_stop_words = stop_words
        self.use_stemmer = stemming
        self.use_lemmitizer = lemmitizer
        self.language = language
        
        self.stemmer = None
        self.lemmitizer = None

        self.prepare()

    def prepare(self) -> None:

        downloader = Downloader()

        # Download dependencies for tokenizer.
        if not downloader.is_installed("punkt"):
            downloader.download("punkt")

        # Download stop words list.
        if self.use_stop_words:
            if not downloader.is_installed("stopwords"):
                downloader.download("stopwords")

        # Download wordnet if needed
        if self.use_lemmitizer:
            if not downloader.is_installed("wordnet"):
                downloader.download("wordnet")
            self.lemmitizer = WordNetLemmatizer()

        if self.use_stemmer:
            self.stemmer = SnowballStemmer(self.language)

    def preprocess_text(self, text):
        
        # Remove punctuations
        text = text.translate(str.maketrans("", "", string.punctuation))

        tokenized_text = word_tokenize(text)
        
        if self.use_stop_words:
            tokenized_text = [word for word in tokenized_text if not word.lower() in stopwords.words()]
        
        if self.use_stemmer: 
            tokenized_text = [self.stemmer.stem(word) for word in tokenized_text]
        
        if self.use_lemmitizer:
            tokenized_text = [self.lemmitizer.lemmatize(word) for word in tokenized_text]


        return tokenized_text

    def compute_overlap_score(self, text1, text2):
        shortest_text = None
        longest_text = None
        count_words =  0

        if len(text1) < len(text2):
            shortest_text = text1
            longest_text = text2
        else:
            shortest_text = text2
            longest_text = text1

        clean_longest = (" ").join(longest_text)

        for word in shortest_text:
            count_words += clean_longest.count(word)

        return count_words/min(len(text1),len(text2))

    def preprocess_arg(self, arg):
        return [arg[1].argId, self.preprocess_text(arg[1].argument)]
    
    def preprocess_kp(self, kp):
        return [kp[1].keyPointId, self.preprocess_text(kp[1].key_point)]

