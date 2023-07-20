import tensorflow as tf

from tensorflow import keras
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Embedding
from keras.models import Sequential

VOCAB_SIZE = 370106
OUTPUT_DIM = 64
MAX_LENGTH = 64

class Convolution(keras.Model):
    def __init__(self,vocab_size=VOCAB_SIZE, output_dim=OUTPUT_DIM, max_length=MAX_LENGTH, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, output_dim, input_length=max_length))
        self.model.add(Conv1D(filters=32, kernel_size=3, activation='relu'), padding='same')
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.2))
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu'), padding='same')
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        
    def call(self, input, training=True):
        return self.model(input)