from transformers import BertTokenizer
from track_1_kp_matching import *

import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tokenizer.tokenizerLF import *
from tokenizer.tokenizerLF import tokenize_LF


class Argument:

    def __init__(self) -> None:
        self.__argId = None
        self.__group = None
        self.__gindex = None
        self.__argument = None
        self.__topic = None
        self.__stance = None

    @property
    def argId(self):
        return self.__argId
    
    @property
    def group(self):
        return self.__group
    
    @property
    def gindex(self):
        return self.__gindex
    
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
    
    @group.setter
    def group(self, group):
        self.__group = group
        
    @gindex.setter
    def gindex(self, gindex):
        self.__gindex = gindex
        
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
        self.__group = None
        self.__gindex = None
        self.__key_point = None 
        self.__topic = None 
        self.__stance = None 
    
    @property
    def keyPointId(self):
        return self.__keyPointId
    
    @property
    def group(self):
        return self.__group
    
    @property
    def gindex(self):
        return self.__gindex
    
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
        
    @group.setter
    def group(self, group):
        self.__group = group
    
    @gindex.setter
    def gindex(self, gindex):
        self.__gindex = gindex
        
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
        # self.__group = None
        # self.__gindex = None
        self.__keyPointId = None
        self.__label = None
        
    @property
    def keyPointId(self):      
        return self.__keyPointId
        
    # @property
    # def group(self):
    #     return self.__group
    
    # @property
    # def gindex(self):
    #     return self.__gindex
    
    @property
    def argId(self):
        return self.__argId
    
    @property
    def label(self):
        return self.__label
    
    @keyPointId.setter
    def keyPointId(self, keyPointId):
        self.__keyPointId = keyPointId
        
    # @group.setter
    # def group(self, group):
    #     self.__group = group
    
    # @gindex.setter
    # def gindex(self, gindex):
    #     self.__gindex = gindex
        
    @argId.setter
    def argId(self, argId):
        self.__argId = argId
        
    @label.setter
    def label(self, label):
        self.__label = label
    
    
class Data():
    def __init__(self) -> None:
        self.__training_data = None
        self.__training_label = None
        self.__validation_data = None
        self.__validation_label = None
        self.__test_data = None
        self.__test_label = None
        
        self.__tokenizer = None
        
    #region getter/setter
    @property
    def training_data(self):
        return self.__training_data
    
    @training_data.setter
    def training_data(self, training_data):
        self.__training_data = training_data
        
    @property
    def training_label(self):
        return self.__training_label
    
    @training_label.setter
    def training_label(self, training_label):
        self.__training_label = training_label
        
    @property
    def validation_data(self):
        return self.__validation_data
    
    @validation_data.setter
    def validation_data(self, validation_data):
        self.__validation_data = validation_data
        
    @property
    def validation_label(self):
        return self.__validation_label
    
    @validation_label.setter
    def validation_label(self, validation_label):
        self.__validation_label = validation_label
        
    @property
    def test_data(self):
        return self.__test_data
    
    @test_data.setter
    def test_data(self, test_data):
        self.__test_data = test_data
        
    @property
    def test_label(self):
        return self.__test_label
    
    @test_label.setter
    def test_label(self, test_label):
        self.__test_label = test_label
        
    @property
    def tokenizer(self):
        return self.__tokenizer;
    @tokenizer.setter
    def tokenizer(self, tokenizer):
        self.__tokenizer = tokenizer

    #endregion 
    
    def readCSV(self, path, subset):
        """read arguments, keypoints and label files
        """
        arguments_file = os.path.join(path, f"arguments_{subset}.csv")
        key_points_file = os.path.join(path, f"key_points_{subset}.csv")
        labels_file = os.path.join(path, f"labels_{subset}.csv")
        
        arguments_df = pd.read_csv(arguments_file)
        key_points_df = pd.read_csv(key_points_file)
        labels_file_df = pd.read_csv(labels_file)

        return arguments_df, key_points_df, labels_file_df
    
    def build_corpus(self, path="kpm_data", subset="train", outFile = "./my_corpus"):
        """build the KPM corpus

        Args:
            path (str): path of the file to be constructed the corpus
            subset (str): subset of the file (train or dev)
            outFile(str): output where to store thhe corpus
    
        """
        print("building corpus...", end="")
        arguments_df, key_points_df, _ = self.readCSV(path, subset)#load_kpm_data(path, subset)
        with open(outFile, "w") as file:
            arg_cols = {}
            for i, col in enumerate(arguments_df.columns):
                arg_cols[col] = i
            for row in arguments_df.to_numpy().tolist():
                file.write(row[arg_cols["argument"]] + ", " + row[arg_cols["topic"]]) 
            
            keyPoints_cols = {}
            for i, col in enumerate(key_points_df.columns):
                keyPoints_cols[col] = i
            for row in key_points_df.to_numpy().tolist():
                file.writelines(row[keyPoints_cols['key_point']] + ", " + row[keyPoints_cols['topic']])
        
        if os.path.exists(outFile):
            print("done!")
            print(f"saved in: {os.path.abspath(outFile)}")
        else:
            print("error")
    
    def process_df(self, df, class_type):
        """create a map with Id as key if the class is 'argument' or 'keypoint', 
        create a list if class is 'label'

        Args:
            df (Dataframe): dataframe to be mapify
            class_type (str): can be only: a (argument), k (key point) or l (label)

        Returns:
            dictionary : the dictionary containing the pairs <ID, class_type>
            or
            list : a list of label
        """
        container = {}
        
        if class_type == 'a':
            for row in df.to_numpy().tolist():
                arg = Argument()
                arg.argId = row[0] #arg_id
                _, arg.group, arg.gindex = arg.argId.split('_')
                arg.argument = row[1] #arguments
                arg.topic = row[2] #topic
                arg.stance = row[3] #stance
                container[arg.argId] = arg
        elif class_type == 'k':
            for row in df.to_numpy().tolist():
                kp = KeyPoint()
                kp.keyPointId = row[0] #key_point_id
                _, kp.group, kp.gindex = kp.keyPointId.split("_")
                kp.key_point = row[1] #key_point
                kp.topic = row[2] #topic
                kp.stance = row[3] #stance
                container[kp.keyPointId] = kp
        elif class_type == 'l':
            container = []
            for row in df.to_numpy().tolist():
                l = Label()
                l.argId = row[0] #arg_id
                l.keyPointId = row[1] #key_point_id
                l.label = row[2] #label
                container.append(l)
        else:
            raise ValueError("argument class_type can be only 'a' (argument), 'k' (key point) or 'l' (label) ")
        
        return container
    
    def make_triple(self):
        pass
    
    def get_data_from(self, path="kpm_data", subset="train"):
        """read csv data from path and the file format should be ./path/filename_{subset}.csv
        filename should be in [arguments, key_points, labels]
        subset should be in [train, dev, test]
        """
        arguments_df, key_points_df, labels_file_df = self.readCSV(path, subset)#load_kpm_data(path, subset)

        labels = self.process_df(labels_file_df, 'l')
        arguments = self.process_df(arguments_df, 'a')
        keyPoints = self.process_df(key_points_df, 'k')

        #tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        
        vocab = load_vocab_file()
        tokenized_data = []
        tokenized_mask = []
        tokenized_attention = []
        maxLen = 10
        
        printfrase = ""
        a=0
        data = []
        targets = []

        for label in labels:
            argId = label.argId
            keyId = label.keyPointId
            if arguments[argId].stance == keyPoints[keyId].stance:
                #data.append('[CLS] '+arguments[argId].argument + ' [SEP] '+ keyPoints[keyId].key_point)
                to_tokenize = '[CLS] '+arguments[argId].argument + ' [SEP] '+ keyPoints[keyId].key_point
                targets.append(label.label)
                vocab, tkz, mask = tokenize_LF(toTokenize=to_tokenize,dictionary=vocab)
                if a == 0:
                    printfrase = to_tokenize
                    a+=1
                
                if len(tkz) > maxLen:
                    maxLen = len(tkz) 
                
                tokenized_data.append(tkz)
                tokenized_mask.append(mask)

        tokenized_data, tokenized_attention = padArray(tokenized_data, maxLen,0,True)
        tokenized_mask = padArray(tokenized_mask, maxLen, 1)
        
        print("length of {} data: {}" .format(subset, str(len(tokenized_data))))
        print("length of {} mask: {}".format(subset, str(len(tokenized_mask))))
        print("length of {} attention: {}".format(subset, str(len(tokenized_attention))))
        print("length of {} labels: {}".format(subset, str(len(targets))))

        #print("original phrase ", printfrase)
        #print("frase ",tk_to_phrase(tokenized_data[0]))
        print("data", tokenized_data[0])
        print("mask", tokenized_mask[0])
        print("att", tokenized_attention[0])
        #tokenized_data.append(tokenized_mask)
        #tokenized_data = tokenizer(data, padding=True, truncation=True)
        
        #tokenized = dict(zip(["input_ids","attention_mask","token_type_ids"],[tokenized_data, tokenized_attention, tokenized_mask]))
        tokenized = {"input_ids":tokenized_data, "attention_mask":tokenized_attention, "token_type_ids":tokenized_mask}
        
        print("here 2")
        
        tokenized_data = tf.data.Dataset.from_tensor_slices(tokenized, targets)
        print("here 3")
        
        if subset == 'train':
            self.training_data = tokenized_data
            self.training_label = targets
        elif subset == 'dev':
            self.validation_data = tokenized_data
            self.validation_label = targets
        elif subset == 'test':
            self.test_data = tokenized_data
            self.test_label = targets
    
