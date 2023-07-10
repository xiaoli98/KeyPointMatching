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

# to use the unipi server for training
#tf.config.gpu.set_per_process_memory_fraction(0.50)
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
    
def oversampling(x, y):
    """oversampling of the dataset, the positive class is oversampled to match the negative class

    Args:
        x (np.array): features
        y (np.array): labels

    Returns:
        x,y: oversampled dataset, concatenation of positives with negatives, need shuffling
    """
    bool_labels = y != 0
    pos = x[bool_labels]
    neg = x[~bool_labels]
    pos_labels = y[bool_labels]
    neg_labels = y[~bool_labels]
    ids = np.arange(len(pos))
    choices = np.random.choice(ids, len(neg))
    res_pos_features = pos[choices]
    res_pos_labels = pos_labels[choices]
    x = np.concatenate([res_pos_features, neg])
    y = np.concatenate([res_pos_labels, neg_labels])
    return x, y

def main():
    useGPU()
    #per usare una sola gpu del server unipi DEVE ESERE COSÌ PER LE REGOLE DI UNIPI
    #os.environ["CUDA_VISIBLE_DEVICES"]= "1" 
    data = dataPreprocess.Data()
    tf_idf_matrix = data.compute_doc_feat_matrix(TfidfVectorizer())
    
    pretrained_models = [
                        # (TFBertModel, "bert-base-uncased", None, None),
                        (TFBertModel, "bert-base-uncased", BertTokenizer, "bert-base-uncased"),
                        (TFBertModel, "bert-large-uncased", BertTokenizer, "bert-large-uncased"),
                        (TFRobertaModel, "roberta-base", RobertaTokenizer, "roberta-base"),
                        (TFRobertaModel, "roberta-large", RobertaTokenizer, "roberta-large"),
                        (TFDistilBertModel, "distilbert-base-uncased", DistilBertTokenizer, "distilbert-base-uncased")
                        ]
    hidden_states=[1]
    
    for model, pretrained, tokenizer, pretrained_tok in pretrained_models:
        for hs in hidden_states:
            
            X_train, y_train, pos, stances = data.create_input(tokenizer=tokenizer,pretrained_tok=pretrained_tok)
    
            X_train = np.array(X_train)
            y_train = np.array(y_train, dtype=np.int32)
            #X_train, y_train = oversampling(X_train, y_train)            
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
            distance_score = tf.keras.Input(1, dtype=tf.float32, name="distance_score")
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
                            epochs=5,
                            batch_size=16,
                            shuffle=True,
                            callbacks=[tensorboard_callback],
                            verbose=1)
            siamese.save(f"models/{pretrained}-{hs}")
            
            data_dev = dataPreprocess.Data(subset="dev")
            tf_idf_matrix_dev = data.compute_doc_feat_matrix(TfidfVectorizer())
            X_train_dev, y_train_dev, pos_dev, stances_dev = data.create_input(tokenizer=tokenizer,pretrained_tok=pretrained_tok)
            X_train_dev = np.array(X_train_dev)
            y_train_dev = np.array(y_train, dtype=np.int32)
            X_train_dev = X_train_dev.reshape(2, len(X_train_dev), INPUT_DIM, MAX_LENGTH)
            
            distance_dev = DistanceLayer(tf_idf_matrix_dev, "cosine")
            distances_dev = []
            for p in tqdm(pos, desc="Precomputing distances"):
                distances_dev.append(distance_dev.compute(p))
            distances_dev = np.array(distances_dev).reshape(len(distances_dev))
            out_dev = siamese.predict(x=(X_train_dev[0], X_train_dev[1], distances_dev))
            with open(f"./prediction_dev_{pretrained}_{hs}.txt", "w") as f:
                for i in range(len(out_dev)):
                    f.write(f"{out_dev[i]}\n")
            
    
if __name__== "__main__":
    main()