import src.model as model
import tensorflow as tf
import src.dataPreprocess as dataPreprocess
import os

import numpy as np

def main():
    data = dataPreprocess.Data()
    dataset = data.get_tf_dataset(n_combinaitons=1, repetition=True)
    
    siamese_model = model.Siamese_Model()
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy')
    # siamese_model.build(dataset)
    # siamese_model.summary()
    print("start training")
    siamese_model.fit(dataset, epochs=10, validation_split=0.3)
    
    
if __name__== "__main__":
    main()