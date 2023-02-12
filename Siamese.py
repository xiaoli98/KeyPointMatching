import numpy as np
import tensorflow as tf

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras
from transformers import TFBertModel as bert

class DistanceLayer(keras.layers.layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    def TF_idf(self, a:str, b:str)->float:
        """compute tf idf score

        Args:
            a (str): first string
            b (str): second string
            
        Returns:
            float: tf idf score
        """
        pass
    
    def jaccard(self, a:str, b:str)->float:
        """compute jaccard similarity

        Args:
            a (str): first string
            b (str): second string

        Returns:
            float: jaccard similarity between a and b
        """
        a = set(a.split()) 
        b = set(b.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    
    def cosine_sim(self, a:str, b:str)->float:
        """compute cosine similarity

        Args:
            a (str): first string
            b (str): second string

        Returns:
            float: cosine similarity between a and b
        """
        vectors = [t for t in self.get_vectors(a,b)]
        return cosine_similarity(vectors)
        
    def call(self, anchor, positive, negative, metric="cosine", vectorizer=None):
        distance = None
        if metric == "cosine":
            distance = self.cosine_sim
        elif metric == "tfidf":
            if vectorizer is None:
                raise ValueError("to use tf-idf you must specify the vectorizer (e.g. provided by sklearn)")
            distance = self.TF_idf
        elif metric == "jaccard":
            metric = self.jaccard
        else:
            raise ValueError("distance metric not found")
        
        ap_distance = distance(anchor, positive)
        an_distance =distance(anchor, negative)
        return (ap_distance, an_distance)

class SiameseBert(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        BertModel = bert.from_pretrained("bert-base-uncased")
        
    def get_vectors(self, *strs):
        text = [t for t in strs]
        vectorizer = CountVectorizer(text)
        vectorizer.fit(text)
        return vectorizer.transform(text).toarray()
        
    def call(self):
        pass