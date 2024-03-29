# import src.model as model
import datetime
import tensorflow as tf
import src.dataPreprocess as dataPreprocess
import os

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from src.Siamese import Siamese
from src.distance import DistanceLayer
from tqdm import tqdm
from transformers import TFBertModel, BertConfig, BertTokenizer
from transformers import TFRobertaModel, RobertaTokenizer
from transformers import TFDistilBertModel, DistilBertTokenizer


MAX_LENGTH = 256
INPUT_DIM = 2
def main():
    data = dataPreprocess.Data()
    tf_idf_matrix = data.compute_doc_feat_matrix(TfidfVectorizer())
    
    pretrained_models = [
                        (TFBertModel, "bert-base-uncased", None, None),
                        (TFBertModel, "bert-base-cased", BertTokenizer, "bert-base-uncased"),
                        (TFBertModel, "bert-large-uncased", BertTokenizer, "bert-base-uncased"),
                        (TFRobertaModel, "roberta-base", RobertaTokenizer, "roberta-base"),
                        (TFRobertaModel, "roberta-large", RobertaTokenizer, "roberta-large"),
                        (TFDistilBertModel, "distilbert-base-uncased", DistilBertTokenizer, "distilbert-base-uncased")
                         ]
    hidden_states=[1, 2, 4]
    
    for model, pretrained, tokenizer, pretrained_tok in pretrained_models:
        for hs in hidden_states:
            
            X_train, y_train, pos, stances = data.create_input(tokenizer=tokenizer,pretrained_tok=pretrained_tok)
    
            X_train = np.array(X_train)
            y_train = np.array(y_train, dtype=np.int32)
            X_train = X_train.reshape(2, len(X_train), INPUT_DIM, MAX_LENGTH)
            
            distance = DistanceLayer(tf_idf_matrix, "cosine")
            distances = []
            for p in tqdm(pos, desc="Precomputing distances"):
                distances.append(distance.compute(p))
            distances = np.array(distances).reshape(len(distances))
            #overlap_baseline = data.overlapping_score()
            #overlap_baseline = np.array(overlap_baseline).reshape(len(overlap_baseline))
            
            log_dir = "logs/fit/" + datetime.datetime.now().strftime(f"%m%d-%H%M-{pretrained}-{hs}")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            
            input1 = tf.keras.Input((INPUT_DIM, MAX_LENGTH), dtype=tf.int32, name="argument")
            input2 = tf.keras.Input((INPUT_DIM, MAX_LENGTH), dtype=tf.int32, name="keypoint")
            distance_score = tf.keras.Input(1, dtype=tf.float32, name="distance score")
           # overlap_score = tf.keras.Input(1, dtype=tf.float32, name="overlap score")

            siamese = Siamese(model=model, pretrained=pretrained, hidden_states_size=hs)
            siamese((input1, input2, distance_score))
            siamese.summary()
           # siamese_out = siamese(input1, input2, distance_score)#, overlap_score)

            #siamese_model = tf.keras.Model(inputs=[input1, input2, distance_score], outputs = siamese_out)
            opt = tf.keras.optimizers.Adam(2e-5)
            loss_fn = tf.keras.losses.BinaryCrossentropy()
            
            siamese.compile(optimizer=opt,
                            loss= loss_fn,
                            metrics=[
                                    tf.keras.metrics.BinaryAccuracy(),
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall()
                                    ]
                        )
            
            print(f"y_train: {np.array(y_train).sum()/len(y_train)}")
            print("start training")
            
            siamese.fit(x=(X_train[0], X_train[1], distances), 
                            y=np.array(y_train), 
                            validation_split = 0.2,
                            epochs=1,
                            batch_size=16,
                            callbacks=[tensorboard_callback],
                            verbose=1)
            siamese.save(f"models/{pretrained}-{hs}")
    
if __name__== "__main__":
    main()