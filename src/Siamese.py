import src.BertLayer as BertLayer

import numpy as np
import tensorflow as tf

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras
from transformers import TFBertModel as bert

class DistanceLayer(keras.layers.Layer):
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

    def get_vectors(self, *strs):
        text = [t for t in strs]
        vectorizer = CountVectorizer(text)
        vectorizer.fit(text)
        return vectorizer.transform(text).toarray()
    
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

class SiameseBert(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = tf.keras.Sequential()
        self.bert = BertLayer(self.classifier)
        self.bert.add(keras.layers.Dense(64, activation = 'relu'))
        self.bert.add(keras.layers.Dropout(0.2))
        self.bert.add(keras.layers.Dense(128, activation = 'relu'))
        self.bert.add(keras.layers.Dense(1, activation = 'sigmoid'))
        
        self.distance = DistanceLayer()
        
        # attributes
        self.metric = None
        for kwarg, value in kwargs.items():
            if kwarg == 'metric':
                self.metric = value
        
    def call(self, X, positive, negative):
        output_x = self.bert(X)
        output_positive = self.bert(positive)
        output_negative = self.bert(negative)
        
        distance = self.distance(output_x, output_positive, output_negative, metric = self.metric)
        
        return distance
        

        