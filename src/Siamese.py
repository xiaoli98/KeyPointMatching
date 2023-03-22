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
        elif metric == "jaccard":
            distance = self.jaccard
        else:
            raise ValueError("distance metric not found")
        
        ap_distance = distance(anchor, positive)
        an_distance =distance(anchor, negative)
        return ap_distance, an_distance

class SiameseBert(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = BertLayer(bert_model_name="bert-base-uncased")
        # creating bert classifier
        self.classifier = keras.Sequential([
            keras.layers.Dense(64, activation = 'relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation = 'relu'),
            keras.layers.Dense(1, activation = 'sigmoid')
        ])
        
        self.distance = DistanceLayer()
        
        # attributes
        self.metric = None
        for kwarg, value in kwargs.items():
            if kwarg == 'metric':
                self.metric = value
            if kwarg == 'decoder':
                self.decoder = value
            
        
    def call(self, X, positive, negative):
        output_x = self.bert(input_id=X.ids, mask=X.attention_mask, type_ids=X.type_ids)
        output_positive = self.bert(input_id=positive.ids, mask=positive.attention_mask, type_ids=positive.type_ids)
        output_negative = self.bert(input_id=negative.ids, mask=negative.attention_mask, type_ids=negative.type_ids)
        
        # distance = self.distance(X, positive, negative, metric = self.metric)
        
        pooled_x = keras.layers.GlobalAveragePooling1D()(output_x)
        pooled_positive = keras.layers.GlobalAveragePooling1D()(output_positive)
        pooled_negative = keras.layers.GlobalAveragePooling1D()(output_negative)
        
        concat_anchor_pos = tf.keras.layers.Concatenate(axis=1)([pooled_x, pooled_positive])
        concat_anchor_neg = tf.keras.layers.Concatenate(axis=1)([pooled_x, pooled_negative])
        # concatenated = tf.keras.layers.Concatenate(axis=1)([pooled_x, pooled_positive, pooled_negative, distance])
        
        return self.classifier(concat_anchor_pos), self.classifier(concat_anchor_neg)
        