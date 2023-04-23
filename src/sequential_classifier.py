import tensorflow as tf
import transformers

from tensorflow import keras
from transformers import TFBertModel

class sequential_classifier(keras.Model):
    def __init__(self, model = TFBertModel, pretrained="bert-base-uncased", hidden_states_size=4, *args, **kwargs):
        super(sequential_classifier, self).__init__(*args, **kwargs)
        
        self.transformer = model.from_pretrained(pretrained, output_hidden_states=True)
        self.transformer.trainable = False
        
        self.concatenate_hidden_states = keras.layers.Concatenate(axis=1)
        self.pooler1 = keras.layers.GlobalAveragePooling1D()
        self.concatenate_output = keras.layers.Concatenate(axis=1)
       
        
        self.dense1 = tf.keras.layers.Dense(1024, activation = 'relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(2048, activation = 'relu')
        self.classifier = tf.keras.layers.Dense(1, activation = 'sigmoid')
        
        
        self.hidden_states_size = hidden_states_size
        
    def call(self, input, training=False):
        
        X1, distance = input
        distance = tf.reshape(distance, [tf.shape(distance)[0], 1])
        
        ids1, attention_mask1 = tf.unstack(X1, axis=1)
        output_x1 = self.transformer(input_ids = ids1, attention_mask=attention_mask1)
        hiddes_states_ind = list(range(-self.hidden_states_size, 0, 1))
        selected_hiddes_states1 = self.concatenate_hidden_states(tuple([output_x1.hidden_states[i] for i in hiddes_states_ind]))
        
        pooled_x1 = self.pooler1(selected_hiddes_states1)    
        concat = self.concatenate_output([pooled_x1, distance])
        
        out = self.dense1(concat)
        if training:
            out = self.dropout(out)
        out = self.dense2(out)
        
        return self.classifier(out)   