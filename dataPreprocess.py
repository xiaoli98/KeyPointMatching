from track_1_kp_matching import *

import os
import pandas as pd


#from transformers import AutoTokenizer


#TODO
#la tokenizzazione si fa dopo aver creato il dataset
#il dataset e' composto da fare una join tra arg_id e key_id presente nel label_file
#

#implementare modelli tipo distillated BERT/BERT_based 


class Argument:

    def __init__(self) -> None:
        self.__argId = None
        self.__argument = None
        self.__topic = None
        self.__stance = None

    @property
    def argId(self):
        return self.__argId
    
    @property
    def argument(self):
        return self.__argument
    
    @property
    def topic(self):
        return self.__topic
    
    @property
    def stance(self):
        return self.__stance
    
    @argId.setter
    def argId(self, argId):
        self.__argId = argId
        
    @argument.setter
    def argument(self, argument):
        self.__argument = argument
        
    @topic.setter
    def topic(self, topic):
        self.__topic = topic
        
    @stance.setter
    def stance(self, stance):
        self.__stance = stance
        

class KeyPoint:
    
    def __init(self) -> None:
        self.__keyPointId = None 
        self.__key_point = None 
        self.__topic = None 
        self.__stance = None 
    
    @property
    def keyPointId(self):
        return self.__keyPointId
    
    @property
    def key_point(self):
        return self.__key_point
    
    @property
    def topic(self):
        return self.__topic
    
    @property
    def stance(self):
        return self.__stance
    
    @keyPointId.setter
    def keyPointId(self, keyPointId):
        self.__keyPointId = keyPointId
        
    @key_point.setter
    def key_point(self, key_point):
        self.__key_point = key_point
        
    @topic.setter
    def topic(self, topic):
        self.__topic = topic
        
    @stance.setter
    def stance(self, stance):
        self.__stance = stance
        

class Label():

    def __init__(self):
        self.__argId = None
        self.__keyPointId = None
        self.__label = None
        
    @property
    def keyPointId(self):      
        return self.__keyPointId
        
    
    @property
    def argId(self):
        return self.__argId
    
    @property
    def label(self):
        return self.__label
    
    @keyPointId.setter
    def keyPointId(self, keyPointId):
        self.__keyPointId = keyPointId
        
        
    @argId.setter
    def argId(self, argId):
        self.__argId = argId
        
    @label.setter
    def label(self, label):
        self.__label = label
        

def preprocess(path="kpm_data", subset="train"):
    arguments_df, key_points_df, labels_file_df = readCSV(path, subset)#load_kpm_data(path, subset)

    labels = []
    for _, row in labels_file_df.iterrows():
        l = Label()
        l.keyPointId = row["key_point_id"]
        l.argId = row["arg_id"]
        l.label = row["label"]
        labels.append(l)

    arguments = {}
    for _, row in arguments_df.iterrows():
        arg = Argument()
        arg.argId = row["arg_id"]
        arg.argument = row["argument"]
        arg.topic = row["topic"]
        arg.stance = row["stance"]
        arguments[arg.argId] = arg 
   
    keyPoints = {}
    for _, row in key_points_df.iterrows():
       kp = KeyPoint()
       kp.keyPointId = row['key_point_id']
       kp.key_point = row['key_point']
       kp.topic = row['topic']
       kp.stance = row['stance']
       keyPoints[kp.keyPointId] = kp
        
    
    processedData = pd.DataFrame()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    data = []
    
    for label in labels:
        argId = label.argId
        keyId = label.keyPointId
        if arguments[argId].stance == keyPoints[keyId].stance:
            tokenized_data = tokenizer('[CLS]' + arguments[argId].argument + '[SEP]'+ keyPoints[keyId].key_point + '[SEP]', return_tensors="np")
            data.append([
                argId, keyId, tokenized_data, label.label
            ])

    return data
    
def readCSV(path, subset):
    arguments_file = os.path.join(path, f"arguments_{subset}.csv")
    key_points_file = os.path.join(path, f"key_points_{subset}.csv")
    labels_file = os.path.join(path, f"labels_{subset}.csv")
       
    arguments_df = pd.read_csv(arguments_file)
    key_points_df = pd.read_csv(key_points_file)
    labels_file_df = pd.read_csv(labels_file)

    return arguments_df, key_points_df, labels_file_df

def getRow(df, idToSearch, cl_name):
    for index, row in df.iterrows():
        if(row[cl_name] == idToSearch):
            return row
    return -1
    

import nltk
from nltk import word_tokenize as tokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer as stemmer
import string

def token(toTokanize, dictionary):
    newDic = dictionary
    
    words_toID = []
    stop_words = stopwords.words('english')
    porter = stemmer()

    filtered_words = toTokanize.lower() #to lower case
    filtered_words = "".join([char for char in filtered_words if char not in string.punctuation]) # remove punctuation 
    filtered_words = tokenizer(filtered_words, "english")   # tokenize
    filtered_words = [word for word in filtered_words if word not in stop_words] # remove stopwords (try without removing during training)
    filtered_words = [porter.stem(word) for word in filtered_words] #stemming
    #print(toTokanize)
    #print(filtered_words)

    for word in filtered_words:
        for w_in, index in newDic.items():
            if w_in == word:
                words_toID.append(index)
                break
        _, index = list(newDic.items())[-1]
        newDic[word] = index + 1
        words_toID.append(index + 1)
    return newDic, words_toID

#preprocess()

dic = {"[CLS]":0,"[SEP]":1}
toTk = "[CLS] On a windy winter morning, a woman looked out of the window.The only thing she saw, a garden. A smile spread across her face as she spotted Maria, her daughter, in the middle of the garden enjoying the weather. It started drizzling. Maria started dancing joyfully.She tried to wave to her daughter, but her elbow was stuck, her arm hurt, her smile turned upside down. Reality came crashing down as the drizzle turned into a storm. Maria's murdered corpse consumed her mind.On a windy winter morning, a woman looked out of the window of her jail cell. [SEP]"
dic, fr = token(toTk, dic)
print(dic)
print("\n")
print(fr)
print("---")
toTk = " [SEP] The schoolboy squirmed. Another two minutes? He knew he should stand at attention. The drillmaster's cane loomed large.Vindhya Himachal … He grunted in discomfort. This was unbearable. He considered making a dash; after all he was in the last row. What if the master noticed? The cane loomed again. He gritted his teeth. Tava shubha … This is it. He cast his eyes around.Jaya he …He started running.Jaya he …He was almost there.Jaya he … The chorus floated from afar. He was already in the toilet, heaving a relieved sigh."
dic, fr = token(toTk, dic)
print(dic)
print("\n")
print(fr)