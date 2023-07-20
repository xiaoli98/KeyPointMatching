import tensorflow as tf

from tensorflow import keras
from keras.layers import Embedding, Dense, LSTM, Bidirectional
from keras.models import Sequential

VOCAB_SIZE = 370106
OUTPUT_DIM = 64
MAX_LENGTH = 64

class LSTMModel(keras.Model):
    
    def __init__(self, vocab_size=VOCAB_SIZE, output_dim=OUTPUT_DIM, max_length=MAX_LENGTH, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, output_dim, input_length=max_length))
        self.model.add(LSTM(64, dropout=0.1))
        self.model.add(LSTM(128, dropout=0.1))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        
    def call(self, input, training=True):
        return self.model(input)
    
class BidirectionalLSTMModel(keras.Model):
    
    def __init__(self, vocab_size=VOCAB_SIZE, output_dim=OUTPUT_DIM, max_length=MAX_LENGTH):
        super().__init__()
        
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, output_dim, input_length=max_length))
        self.model.add(Bidirectional((LSTM(64, dropout=0.1))))
        self.model.add(Bidirectional(LSTM(128, dropout=0.1)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        
    def call(self, input, training=True):
        return self.model(input)