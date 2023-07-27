import concurrent.futures
import string
from collections import Counter

import src.dataPreprocess as dataPreprocess
from multiprocessing import Pool 

from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense

import numpy as np
import re

from gensim.corpora import Dictionary

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams 
  
nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')

correct_words = words.words()
# Get the list of English stopwords and stemmer
stop_words = set(stopwords.words('english'))

not_in_corpora = {"haaapy":0,"sada":0}

def tokenize_LSTM(path="kpm_data", subset="train", pad_to_length=None):
    
    data = dataPreprocess.Data(path=path, subset=subset)

    labels = data.process_df(data.label_df, 'l')
    arguments = data.process_df(data.arguments_df, 'a')
    keyPoints = data.process_df(data.key_points_df, 'k')

    corpus = load_vocab_file()
    dct = Dictionary(corpus)
    
    for label in tqdm(labels, desc="-Building vocab-"):
        #in realtà stiamo facendo solo un concat di ciò che dobbiamo tokenizzare, la tokenizzazione effettiva avviene più aventi nel codice
        #questo ci serve per fare l'operazione di concat una sola volta e creare il vocab
        label.tokenized = (arguments[str(label.argId)].argument+" "+keyPoints[str(label.keyPointId)].key_point)
        
    """for label in tqdm(labels):
        check_tokenization_errors(label, dct)

    print("all unknown words")
    print(not_in_corpora)    

    wrong_w = list(not_in_corpora.keys())
    correct_w(wrong_w)

    input()"""
    #with Pool() as pool:
    #    labels = list(tqdm(pool.starmap(preprocess_text, [(label, dct) for label in labels]), desc="-Token to ids-"))

    for label in tqdm(labels, desc = "- Tokens to ids -"):
        label = preprocess_text(label, dct)

    sequence = [label.tokenized for label in labels]
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post', maxlen=pad_to_length)
    print("shape x: ", sequence.shape)
    #sequence = np.reshape(sequence,(sequence.shape[0] ,1, sequence.shape[1]))
    #print("shape x: ", sequence.shape)
    
    x = sequence#sequence.tolist()
    y = [label.label for label in labels]
    
    return x, y, len(dct.keys())

    

    """print("all unknown words")
    print(not_in_corpora)
    input()
    counter = 0
    negone = 0
    for label in tqdm(labels):
        for w in label.tokenized:
            counter +=1
            if w == -1:
                negone +=1
    
    print("found -1 ",negone," out of ",counter)
    input()"""

def preprocess_text(label, dct):
    tokens = reg_text_clean(label.tokenized)
    label.tokenized = [token.lower() for token in tokens if token.lower() not in stop_words]
    label.tokenized = dct.doc2idx(label.tokenized, 0)
    return label

def load_vocab_file():
    corpus = [[""]]
    with open('words_alpha.txt') as word_file:
        valid_words = set(word_file.read().split())
    
    for word in tqdm(valid_words, desc = "-buinding corpus-"):
        corpus.append([word])
    return corpus

def check_padding_errors(labels):
    size = len(labels[0].tokenized)
    error = 0
    for label in labels:
        if len(label.tokenized) != size:
            error +=1
    
    print("there are ",error," errors in size")

def check_tokenization_errors(label, dct):
    tokens = reg_text_clean(label.tokenized)
    label.tokenized = [token.lower() for token in tokens if token.lower() not in stop_words]

    temp_ph = label.tokenized
    label.tokenized = dct.doc2idx(label.tokenized)

    count = 0
    for val in label.tokenized:
        if val == -1:
            word = temp_ph[count]
            if word in not_in_corpora:
                not_in_corpora[word] += 1
            else: 
                not_in_corpora[word] = 1
        count += 1

    return label

def remove_punctuation(text):
    # Use regex to remove punctuation
    return re.sub(r'[^\w\s]', ' ', text)

def separate_slash_words(text):
    # Replace slash-separated words with a space
    return re.sub(r'/', ' ', text)

def remove_numbers(text):
    # Use regex to remove numbers
    return re.sub(r'\d+', ' ', text)

def remove_contractions(text):
    # List of common contractions
    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": "not",
        "'s": "",
        "'ll": "will",
        "'ve": "have",
        "'re": "are",
        "'d": "would",
        "'m": "am"
    }
    
    # Tokenize the text and remove contractions
    words = word_tokenize(text)
    words = [contractions[word] if word in contractions else word for word in words]
    return words

def reg_text_clean(text):
    # Separate words joined by "/"
    text = separate_slash_words(text)

    # Remove punctuation
    text = remove_punctuation(text)
    
    #Remove numbers
    text = remove_numbers(text)

    # Remove contractions
    text = remove_contractions(text)
    return text

def correct_w(incorrect_words):
    # list of incorrect spellings
    # that need to be corrected
    print("----------")
    print(incorrect_words)
    # loop for finding correct spellings
    # based on jaccard distance
    # and printing the correct word
    for word in incorrect_words:
        temp = [(jaccard_distance(set(ngrams(word, 2)),
                                set(ngrams(w, 2))),w)
                for w in correct_words if w[0]==word[0]]
        print(sorted(temp, key = lambda val:val[0]).pop(0))


#print("starting")
#print(tokenize_LSTM()[0].tokenized)
