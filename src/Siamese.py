import tensorflow as tf
import transformers

from tensorflow import keras
from transformers import TFBertModel


class Siamese(keras.Model):
    
    def __init__(self, model = TFBertModel, pretrained="bert-base-uncased", hidden_states_size=4, *args, **kwargs):
        super(Siamese, self).__init__(*args, **kwargs)
        
        self.transformer = model.from_pretrained(pretrained, output_hidden_states=True)
        #FIXME this is a hack to make the model faster
        self.transformer.trainable = True
        
        self.concatenate_hidden_states = keras.layers.Concatenate(axis=1)
        self.pooler1 = keras.layers.GlobalAveragePooling1D()
        self.concatenate_output = keras.layers.Concatenate(axis=1)
       
        self.sequential = tf.keras.Sequential()
        self.sequential.add(tf.keras.layers.Dense(1024, tf.keras.activations.relu))
        self.sequential.add(tf.keras.layers.Dropout(0.2))
        self.sequential.add(tf.keras.layers.Dense(512, tf.keras.activations.relu))
        self.sequential.add(tf.keras.layers.Dropout(0.2))
        self.sequential.add(tf.keras.layers.Dense(256, tf.keras.activations.relu))
        self.sequential.add(tf.keras.layers.Dropout(0.2))
        self.sequential.add(tf.keras.layers.Dense(128, tf.keras.activations.relu))
        self.sequential.add(tf.keras.layers.Dense(1, tf.keras.activations.sigmoid))

        self.hidden_states_size = hidden_states_size
        
    def summary(self):
        super().summary()
        self.sequential.summary()
        
    def call(self, input, training=True):
        X1, X2, distance = input
        distance = tf.reshape(distance, [tf.shape(distance)[0], 1])
        
        ids1, attention_mask1 = tf.unstack(X1, axis=1)
        ids2, attention_mask2 = tf.unstack(X2, axis=1)
        
        output_x1 = self.transformer(input_ids = ids1, attention_mask=attention_mask1)
        output_x2 = self.transformer(input_ids = ids2, attention_mask=attention_mask2)
        
        hiddes_states_ind = list(range(-self.hidden_states_size, 0, 1))
        selected_hiddes_states1 = self.concatenate_hidden_states(tuple([output_x1.hidden_states[i] for i in hiddes_states_ind]))
        selected_hiddes_states2 = self.concatenate_hidden_states(tuple([output_x2.hidden_states[i] for i in hiddes_states_ind]))
    
        
        pooled_x1 = self.pooler1(selected_hiddes_states1)
        pooled_x2 = self.pooler1(selected_hiddes_states2)
        
        concat = self.concatenate_output([pooled_x1, pooled_x2, distance])
        
        output = self.sequential(concat)
        
        return output      