# import src.BertLayer as BertLayer

import tensorflow as tf
import transformers

from tensorflow import keras
from transformers import TFBertModel as bert

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
        
        # attributes
        self.metric = None
        for kwarg, value in kwargs.items():
            if kwarg == 'metric':
                self.metric = value
            if kwarg == 'decoder':
                self.decoder = value
            
        
    def call(self, X1, X2, distance, training=False):
        ids1, attention_mask1, token_type_ids1 = tf.unstack(X1, axis=1)
        ids2, attention_mask2, token_type_ids2 = tf.unstack(X2, axis=1)
        
        output_x1 = self.bert(input_ids = ids1, token_type_ids=token_type_ids1, attention_mask=attention_mask1)
        output_x2 = self.bert(input_ids = ids2, token_type_ids=token_type_ids2, attention_mask=attention_mask2)
        
        hidden_states_size = 4 # count of the last states 
        hiddes_states_ind = list(range(-hidden_states_size, 0, 1))
        selected_hiddes_states1 = self.concatenate_hidden_states(tuple([output_x1.hidden_states[i] for i in hiddes_states_ind]))
        selected_hiddes_states2 = self.concatenate_hidden_states(tuple([output_x2.hidden_states[i] for i in hiddes_states_ind]))
        
        pooled_x1 = self.pooler1(selected_hiddes_states1)
        pooled_x2 = self.pooler2(selected_hiddes_states2)

        concat = self.concatenate_output([pooled_x1, pooled_x2, distance])

        out = self.dense1(concat)
        if training:
            out = self.dropout(out)
        out = self.dense2(out)
        return self.classifier(out)        