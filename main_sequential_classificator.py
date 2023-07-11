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


from tensorflow.python.framework.config import set_memory_growth
def useGPU():
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.set_visible_devices(physical_devices[1], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[1], True)
        tf.config.set_visible_devices(physical_devices[2], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[2], True)
    except:
     #Invalid device or cannot modify virtual devices once initialized.
        pass
    #os.environ['CUDA_HOME'] = '/usr/local/cuda'
    #os.environ['PATH']= '/usr/local/cuda/bin:$PATH'  
    #os.environ['CPATH'] = '/usr/local/cuda/include:$CPATH'  
    #os.environ['LIBRARY_PATH'] = '/usr/local/cuda/lib64:$LIBRARY_PATH'  
    #os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH'  
    #os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:$LD_LIBRARY_PATH'
    #os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def main():
    useGPU()
    
    data = dataPreprocess.Data()
    tf_idf_matrix = data.compute_doc_feat_matrix(TfidfVectorizer())
    #feature_matrix = data.compute_doc_feat_matrix(CountVectorizer())
    
    pretrained_models = [
                        #(TFBertModel, "bert-base-uncased", None, None),
                        (TFBertModel, "bert-base-uncased", BertTokenizer, "bert-base-uncased"),
                        (TFRobertaModel, "roberta-base", RobertaTokenizer, "roberta-base"),
                        (TFDistilBertModel, "distilbert-base-uncased", DistilBertTokenizer, "distilbert-base-uncased"),
                        (TFBertModel, "bert-large-uncased", BertTokenizer, "bert-large-uncased"),
                        (TFRobertaModel, "roberta-large", RobertaTokenizer, "roberta-large")
                         ]
    hidden_states=[1]
    
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
            distance_score = tf.keras.Input(1, dtype=tf.float32, name="distance_score")
            #overlap_score = tf.keras.Input(1, dtype=tf.float32, name="overlap_score")

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
                            epochs=5,
                            batch_size=16,
                            callbacks=[tensorboard_callback],
                            verbose=1)
            
            classifier.save(f"models/{pretrained}-{hs}")

            print("end of training")
        
            data_dev = dataPreprocess.Data(subset="dev")
            tf_idf_matrix_dev = data_dev.compute_doc_feat_matrix(TfidfVectorizer())
            X_dev, y_dev, pos_dev, stances_dev = data_dev.create_input(tokenizer=tokenizer,pretrained_tok=pretrained_tok,using_sq_classifier=True)
    
            X_dev = np.array(X_dev)
            print("X_train: ", X_dev.shape)
            y_dev= np.array(y_dev, dtype=np.int32)
            
            distance_dev = DistanceLayer(tf_idf_matrix_dev, "cosine")
            distances_dev = []

            for p in tqdm(pos_dev, desc="Precomputing distances"):
                distances_dev.append(distance_dev.compute(p))
                #distances_j.append(distance_j.compute(p))
                
            distances_dev = np.array(distances_dev).reshape(len(distances_dev))
            

            out = classifier.predict(x=(X_dev, distances_dev))
            
            with open(("prediction_dev_"+pretrained), "w") as f:
              for o in out:
                f.writelines(str(o))
            
if __name__== "__main__":
    main()