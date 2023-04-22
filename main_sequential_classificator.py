import datetime
import tensorflow as tf
import src.dataPreprocess as dataPreprocess
import os

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from src.sequential_classifier import sequential_classifier
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
    #feature_matrix = data.compute_doc_feat_matrix(CountVectorizer())
    
    pretrained_models = [
                        (TFBertModel, "bert-base-uncased", None, None),
                        #(TFBertModel, "bert-base-cased", BertTokenizer, "bert-base-uncased"),
                        #(TFBertModel, "bert-large-uncased", BertTokenizer, "bert-base-uncased"),
                        #(TFRobertaModel, "roberta-base", RobertaTokenizer, "roberta-base"),
                        #(TFRobertaModel, "roberta-large", RobertaTokenizer, "roberta-large"),
                        #(TFDistilBertModel, "distilbert-base-uncased", DistilBertTokenizer, "distilbert-base-uncased")
                         ]
    hidden_states=[1, 2, 4]
    
    for model, pretrained, tokenizer, pretrained_tok in pretrained_models:
        for hs in hidden_states:
            
            X_train, y_train, pos, stances = data.create_input(tokenizer=tokenizer,pretrained_tok=pretrained_tok,using_sq_classifier=True)
    
            X_train = np.array(X_train)
            print("X_train: ", X_train.shape)
            y_train = np.array(y_train, dtype=np.int32)
            #X_train = X_train.reshape(1, len(X_train), INPUT_DIM, MAX_LENGTH)
            
            distance = DistanceLayer(tf_idf_matrix, "cosine")
            #distance_j = DistanceLayer(feature_matrix, "jaccard")
            distances = []
            #distances_j = []
            for p in tqdm(pos, desc="Precomputing distances"):
                distances.append(distance.compute(p))
                #distances_j.append(distance_j.compute(p))
                
            distances = np.array(distances).reshape(len(distances))
            #distance_j = np.array(distances_j).reshape(len(distances_j))
            
            overlap_baseline = data.overlapping_score()
            overlap_baseline = np.array(overlap_baseline).reshape(len(overlap_baseline))
            
            
            log_dir = "logs/fit/" + datetime.datetime.now().strftime(f"%m%d-%H%M-{pretrained}-{hs}")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            
            input1 = tf.keras.Input((INPUT_DIM, MAX_LENGTH), dtype=tf.int32, name="argument")
            distance_score = tf.keras.Input(1, dtype=tf.float32, name="distance score")
            #overlap_score = tf.keras.Input(1, dtype=tf.float32, name="overlap score")

            classifier = sequential_classifier(model=model, pretrained=pretrained, hidden_states_size=hs)
            classifier((input1, distance_score))
            classifier.summary()
           
            opt = tf.keras.optimizers.Adam(2e-5)
            loss_fn = tf.keras.losses.BinaryCrossentropy()
            
            classifier.compile(optimizer=opt,
                            loss= loss_fn,
                            metrics=[
                                    tf.keras.metrics.BinaryAccuracy(),
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall()
                                    ]
                        )
            
            print(f"y_train: {np.array(y_train).sum()/len(y_train)}")
            print("start training of the sequential_classifier")
            
            classifier.fit(x=(X_train, distances), 
                            y=np.array(y_train), 
                            epochs=1,
                            batch_size=16,
                            callbacks=[tensorboard_callback],
                            verbose=1)
            classifier.save(f"models/{pretrained}-{hs}")
    
if __name__== "__main__":
    main()