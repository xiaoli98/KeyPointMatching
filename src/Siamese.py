# import src.BertLayer as BertLayer

import numpy as np
import tensorflow as tf
import transformers

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
    def __init__(self, bert_model_name="bert-base-uncased", *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        config = transformers.BertConfig()
        config.output_hidden_states = True
        
        self.bert = bert.from_pretrained(bert_model_name, config=config)
        
        self.dense1 = tf.keras.layers.Dense(64, activation = 'relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(128, activation = 'relu')
        self.classifier = tf.keras.layers.Dense(1, activation = 'sigmoid')
        
        self.concatenate_hidden_states = keras.layers.Concatenate(axis=1)
        self.concatenate_output = keras.layers.Concatenate(axis=1)
        self.pooler1 = keras.layers.GlobalAveragePooling1D()
        self.pooler2 = keras.layers.GlobalAveragePooling1D()
        # self.distance = DistanceLayer()

        
        # attributes
        self.metric = None
        for kwarg, value in kwargs.items():
            if kwarg == 'metric':
                self.metric = value
            if kwarg == 'decoder':
                self.decoder = value
            
    def build(self, input_shape):
        
        pass
        
    def call(self, X1, X2, training=False):
        # X1 = X[0]
        # X2 = X[1]
        # print(f"{SiameseBert.call.__qualname__}: values of X is {X}")
        # print(f"len of X: {len(X)}")
        # print(f"{SiameseBert.call.__qualname__}: values of X1 is {X1}")
        # print("-"*100)
        # print(f"{SiameseBert.call.__qualname__}: values of X2 is {X2}")
        # print(f"{SiameseBert.call.__qualname__}: values of X[0] is {X[0]}")
        # print("-"*100)
        # print(f"{SiameseBert.call.__qualname__}: values of X[1] is {X[1]}")
        # print("-"*100)
        # print(f"{SiameseBert.call.__qualname__}: values of X[2] is {X[2]}")
        # print("-"*100)
        # print(f"{SiameseBert.call.__qualname__}: values of X[3] is {X[3]}")
        # print("-"*100)
        # print(f"{SiameseBert.call.__qualname__}: values of X[4] is {X[4]}")
        # print("-"*100)
        # print(f"{SiameseBert.call.__qualname__}: values of X[5] is {X[5]}")
        # print("-"*100)
        # output_x1 = self.bert(input_ids=X1[0], token_type_ids=X1[1], attention_mask=X1[2])
        # output_x2 = self.bert(input_ids=X2[0], token_type_ids=X2[1], attention_mask=X2[2])
        # output_x1 = self.bert(input_ids=tf.cast(X[0], dtype=tf.int32), token_type_ids=tf.cast(X[1], dtype=tf.int32), attention_mask=tf.cast(X[2], dtype=tf.int32))
        # output_x2 = self.bert(input_ids=tf.cast(X[3], dtype=tf.int32), token_type_ids=tf.cast(X[4], dtype=tf.int32), attention_mask=tf.cast(X[5], dtype=tf.int32))
        
        ids1, attention_mask1, token_type_ids1 = tf.unstack(X1, axis=1)
        ids2, attention_mask2, token_type_ids2 = tf.unstack(X2, axis=1)
        
        output_x1 = self.bert(input_ids = ids1, token_type_ids=token_type_ids1, attention_mask=attention_mask1)
        output_x2 = self.bert(input_ids = ids2, token_type_ids=token_type_ids2, attention_mask=attention_mask2)
        # print(f"{SiameseBert.call.__qualname__}: output_x1[0]: {output_x1.hidden_states}")
        # print(f"{SiameseBert.call.__qualname__}: output_x2[0]: {output_x2.hidden_states}")
        # distance = self.distance(X, positive, negative, metric = self.metric)
        
        hidden_states_size = 4 # count of the last states 
        hiddes_states_ind = list(range(-hidden_states_size, 0, 1))
        selected_hiddes_states1 = self.concatenate_hidden_states(tuple([output_x1.hidden_states[i] for i in hiddes_states_ind]))
        selected_hiddes_states2 = self.concatenate_hidden_states(tuple([output_x2.hidden_states[i] for i in hiddes_states_ind]))
        
        pooled_x1 = self.pooler1(selected_hiddes_states1)
        pooled_x2 = self.pooler2(selected_hiddes_states2)
        # pooled_x1 = output_x1.pooler_output
        # pooled_x2 = output_x2.pooler_output
        
        # print(f"{SiameseBert.call.__qualname__}: pooled_x1: {pooled_x1}")
        # print(f"{SiameseBert.call.__qualname__}: pooled_x2: {pooled_x2}")

        concat = self.concatenate_output([pooled_x1, pooled_x2])
        # print("-"*100)
        # print(f"{SiameseBert.call.__qualname__}: values of concat is {concat}")
        # print("-"*100)
        out = self.dense1(concat)
        if training:
            out = self.dropout(out)
        out = self.dense2(out)
        return self.classifier(out)        