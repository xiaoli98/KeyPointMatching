# import src.model as model
import tensorflow as tf
import src.dataPreprocess as dataPreprocess
import os

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from src.Siamese import SiameseBert
from src.distance import DistanceLayer
from tqdm import tqdm

MAX_LENGTH = 128

def main():
    data = dataPreprocess.Data()
    tf_idf_matrix = data.compute_doc_feat_matrix(TfidfVectorizer())
    X_train, y_train, pos = data.create_input()
    
    X_train = np.array(X_train)
    y_train = np.array(y_train, dtype=np.int32)
    X_train = X_train.reshape(2, len(X_train), 3, MAX_LENGTH)
    
    input1 = tf.keras.Input((3,MAX_LENGTH), dtype=tf.int32)
    input2 = tf.keras.Input((3,MAX_LENGTH), dtype=tf.int32)
    distance_score = tf.keras.Input(1, dtype=tf.float32)

    distance = DistanceLayer(tf_idf_matrix, "cosine")
    distances = []
    for p in tqdm(pos, desc="Precomputing distances"):
        distances.append(distance.compute(p))
    distances = np.array(distances).reshape(len(distances))
    
    siamese = SiameseBert()

    siamese_out = siamese(input1, input2, distance_score)

    siamese_model = tf.keras.Model(inputs=[input1, input2, distance_score], outputs = siamese_out)
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                          loss=tf.keras.losses.BinaryCrossentropy(),
                          metrics=["accuracy"])
    siamese_model.summary()
    print("start training")
    siamese_model.fit(x=(X_train[0], X_train[1], distances), 
                      y=np.array(y_train), 
                      shuffle=True,
                      validation_split = 0.2,
                      epochs=2,
                      batch_size=8,
                      verbose=2)
    
    
if __name__== "__main__":
    main()