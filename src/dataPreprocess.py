from src.track_1_kp_matching import *

import os
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from .tokenizer.kpmTokenizer import KPMTokernizer

import random

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from src.overlap_score import * 
from multiprocessing import Pool
from os.path import exists


class Argument:

    def __init__(self) -> None:
        self.__argId = None
        self.__group = None
        self.__gindex = None
        self.__argument = None
        self.__topic = None
        self.__stance = None
        self.__tfidf_pos = None

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

    @property
    def tfidf_pos(self):
        return self.__tfidf_pos
    

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
        
    @tfidf_pos.setter
    def tfidf_pos(self, tfidf_pos):
        self.__tfidf_pos = tfidf_pos

    def printAll(self):
        print("argId:",self.__argId, " group:", self.__group," gindex:", self.__gindex, " argument:",self.__argument, " topic:",self.__topic," stance:",self.__stance)


class KeyPoint:
    
    def __init(self) -> None:
        self.__keyPointId = None 
        self.__group = None
        self.__gindex = None
        self.__key_point = None 
        self.__topic = None 
        self.__stance = None 
        self.__tfidf_pos = None

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
    
    @property
    def tfidf_pos(self):
        return self.__tfidf_pos
    

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
    
    @tfidf_pos.setter
    def tfidf_pos(self, tfidf_pos):
        self.__tfidf_pos = tfidf_pos

    def printAll(self):
        print("key_point_id:",self.__keyPointId," group:",self.__group," gindex:",self.__gindex," key_point:",self.__key_point," topic:",self.__topic," stance:",self.__stance)

class Label():

    def __init__(self):
        self.__argId = None
        # self.__group = None
        # self.__gindex = None
        self.__keyPointId = None
        self.__label = None
        self.__tokenized = None
        
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
    
    @property
    def tokenized(self):
        return self.__tokenized
        
    @argId.setter
    def argId(self, argId):
        self.__argId = argId
        
    @label.setter
    def label(self, label):
        self.__label = label
        
    @tokenized.setter
    def tokenized(self, tokenized):
        self.__tokenized = tokenized
    
    def printAll(self):
        print("argID:",self.__argId, " keyPointId:", self.__keyPointId, " label:",self.__label, " ",self.__tokenized)
    
class Data():
    def __init__(self, path="kpm_data", subset="train") -> None:
        self.__training_data = None
        self.__training_label = None
        self.__validation_data = None
        self.__validation_label = None
        self.__test_data = None
        self.__test_label = None
        
        self.__tokenizer = None
        self.counter = 0
        self.arguments_df, self.key_points_df, self.label_df = self.readCSV(path, subset)
        
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
    
    def build_corpus(self, path ="./src/corpuses", outFile = "/my_corpus"):
        """build the KPM corpus

        Args:
            path (str): path of the file to be constructed the corpus
            subset (str): subset of the file (train or dev)
            outFile(str): output where to store thhe corpus
    
        """
        if os.path.exists(path+outFile):
            return 
        
        print("building corpus...", end="")
        
        if not os.path.exists(path):  
            os.makedirs(path)       
      
        # arguments_df, key_points_df, _ = self.readCSV(path, subset)#load_kpm_data(path, subset)
        with open(path+outFile, "w") as file:
            arg_cols = {}
            for i, col in enumerate(self.arguments_df.columns):
                arg_cols[col] = i
            for row in self.arguments_df.to_numpy().tolist():
                file.write(row[arg_cols["argument"]] + ", " + row[arg_cols["topic"]]) 
            
            keyPoints_cols = {}
            for i, col in enumerate(self.key_points_df.columns):
                keyPoints_cols[col] = i
            for row in self.key_points_df.to_numpy().tolist():
                file.writelines(row[keyPoints_cols['key_point']] + ", " + row[keyPoints_cols['topic']])
        
        if os.path.exists(path+outFile):
            print("done!")
            print(f"saved in: {os.path.abspath(outFile)}")
        else:
            print("error")
            
    def compute_doc_feat_matrix(self, vectorizer):
        content = []
        arg_cols = {}
        for i, col in enumerate(self.arguments_df.columns):
            arg_cols[col] = i
        for row in self.arguments_df.to_numpy().tolist():
            content.append(row[arg_cols["argument"]] + ", " + row[arg_cols["topic"]])
            
        keyPoints_cols = {}
        for i, col in enumerate(self.key_points_df.columns):
            keyPoints_cols[col] = i
        for row in self.key_points_df.to_numpy().tolist():
            content.append(row[keyPoints_cols['key_point']] + ", " + row[keyPoints_cols['topic']])
        
        return vectorizer.fit_transform(content)
    
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
                arg.tfidf_pos = self.counter
                self.counter += 1
                container[arg.argId] = arg
        elif class_type == 'k':
            for row in df.to_numpy().tolist():
                kp = KeyPoint()
                kp.keyPointId = row[0] #key_point_id
                _, kp.group, kp.gindex = kp.keyPointId.split("_")
                kp.key_point = row[1] #key_point
                kp.topic = row[2] #topic
                kp.stance = row[3] #stance
                kp.tfidf_pos = self.counter
                self.counter += 1
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
    
    
   
    def make_siamese_input(self, n_combinaitons = 3, repetition = True, pretrained_tok = None):
        """
            get the anchor from label e.g. arg_0_0,kp_0_0,0 
            than fond a positive example (another label) same topic, same kp when possible 
            and a negtive example, same topic same kp 
            For positive and negative we mean that if 
            the label of the anchor is 1 than positive is 1 and negative is 0    
            if label is 0 than positive is 0 and negative is 1

            e.g
                anchor = arg_0_0,kp_0_0,0
                positive = arg_0_1,kp_0_0,0 where we get the actual values so [arg,kp] 
                negative = arg_0_9,kp_0_0,1
                
            
            n_combinaitons = 3, implies the number of example to find for each label, if it is set to -1 it will generate all the possible combination;
                                if the value is grater than the max possible combinations and repetition can be done the number of combinaitons will be set to max(between positive and negative examples) directly 
                                if the value is grater than the max possible combinations and repetition can't be done the number of combinaitons will be set to min(between positive and negative examples) directly 
            repetition = True, if true implies that the same value ca be used more than once
            
        """

        #Declare the tokenizer using the one we crated
        corpus = "./src/corpuses/"
        print("hello ")
        
        self.build_corpus()
        
        
        self.tokenizer = KPMTokernizer(pretrained="bert-base-cased")
        if pretrained_tok is None:          
            self.tokenizer.train([corpus], "./my_pretrained_bert_tok.tkn")
        else:
            self.tokenizer = KPMTokernizer(pretrained=pretrained_tok)


        #recover the data from the file 
        # arguments_df, key_points_df, labels_file_df = self.readCSV(path, subset)

        labels = self.process_df(self.label_df, 'l')
        arguments = self.process_df(self.arguments_df, 'a')
        keyPoints = self.process_df(self.key_points_df, 'k')
        
        #devide the data based on the group and adding the tokenized argument for argument and
        # keypoint,topic for the topic 
        arg_byGroup = {}
        kp_byGroup = {}

        for _, arg in tqdm(arguments.items()): 
            if arg.group in arg_byGroup:
                arg_byGroup[str(arg.group)].append(arg)
            else:
                arg_byGroup[str(arg.group)] = [arg]

        for _, kp in tqdm(keyPoints.items()):
            if kp.group in kp_byGroup:
                kp_byGroup[str(kp.group)].append(kp)
            else:
                kp_byGroup[str(kp.group)] = [kp]

        #print(labels[0].argId)

        ##dividere le labels based on the group of kp and if they are postitive or negative
        ## labes_byGroup {"kp_id": 0, "positive":[], "negative":[]}
        lb_pos_neg_ByGroup = []
        for label in tqdm(labels):
            
            kp_id = label.keyPointId #keyPoints[str(label.keyPointId)].group    
            label.tokenized = self.tokenizer.encode(arguments[str(label.argId)].argument+" "+arguments[str(label.argId)].topic, keyPoints[str(label.keyPointId)].key_point+" "+keyPoints[str(label.keyPointId)].topic)
            result_dict = next((item for item in lb_pos_neg_ByGroup if item['kp_id'] == kp_id), None)
            
            if result_dict != None:
                if label.label == 1:                     
                    result_dict["negative"].append(label)
                else:
                    result_dict["positive"].append(label)
            else:
                dic = {}
                if  label.label == 1:
                    dic = {"kp_id": kp_id, "positive":[], "negative":[label]}
                else:
                    dic = {"kp_id": kp_id, "positive":[label], "negative":[]}
                
                lb_pos_neg_ByGroup.append(dic)              

        #print(len(lb_pos_neg_ByGroup))
        #print(lb_pos_neg_ByGroup[0]["positive"][0].printAll())
            
        ##assigna a radom pos and negative based on the lable we are analizing in the top for
        an_pos_neg = []
        for label in tqdm(labels):

            kp_id = str(label.keyPointId)          
            result_dict = next((item for item in lb_pos_neg_ByGroup if item['kp_id'] == kp_id), None)
            anchor = label.tokenized
                      
            positive_dic = result_dict["positive"]
            negative_dic = result_dict["negative"]
            
            ##print(len(positive_dic))
            ##print(len(negative_dic))
            n_com = n_combinaitons
            
            if (n_com == -1 or n_com > min(len(positive_dic),len(negative_dic))) and repetition == False:
                n_com = min(len(positive_dic),len(negative_dic))
                
            elif n_com == -1 and repetition == True:
                n_com = max(len(positive_dic),len(negative_dic))
            
            if repetition == False:
                choose_pos = random.sample(range(0,len(positive_dic)-1),n_com-1)
                choose_neg = random.sample(range(0,len(negative_dic)-1),n_com-1)
                
                fromPositive = None
                fromNegative = None
                
                for i in range(0,n_com-1):
                    fromPositive = positive_dic[choose_pos[i]] 
                    fromNegative = negative_dic[choose_neg[i]]

                    positive = []
                    negative = []  
                        
                    if label.label == 1:
                        negative = fromPositive.tokenized
                        positive = fromNegative.tokenized
                    else:
                        positive = fromPositive.tokenized
                        negative = fromNegative.tokenized
                        
                    #print(positive[0].printAll(), positive[1].printAll())
                    #print(negative[0].printAll(), negative[1].printAll())
                    
                    an_pos_neg.append([anchor, positive, negative])
                
            else:
                for i in range (0,n_com):                 
                    fromPositive = positive_dic[random.randint(0,len(positive_dic)-1)]
                    fromNegative = negative_dic[random.randint(0,len(negative_dic)-1)]
      
                    positive = []
                    negative = []  
                        
                    if label.label == 1:
                        negative = fromPositive.tokenized
                        positive = fromNegative.tokenized
                    else:
                        positive = fromPositive.tokenized
                        negative = fromNegative.tokenized
                        
                    #print(positive[0].printAll(), positive[1].printAll())
                    #print(negative[0].printAll(), negative[1].printAll())
                    
                    an_pos_neg.append([[anchor, positive, negative], label.label])                    
        return an_pos_neg
 
 
    def create_input(self, tokenizer=None, pretrained_tok = None, corpus = "./src/corpuses/my_corpus", using_sq_classifier = False):
        """create the input data for siamese model

        Args:
            pretrained_tok (Tokenizer, optional): a pretrained tokenizer. Defaults to None.
            corpus (str, optional): path to a corpus. Defaults to "./src/corpuses/my_corpus".
            using_sq_classifier: if true returns the input for the sequential classifier

        Returns:
            (X, y, pos): 
                -   X are the data for siamese model
                -   y are labels
                -   pos is the position of the data, it is a tuple of tow elements, one for arguments and one for key points 
        """
        
        using_batch_encoding = False    #this flag is for distinguishing huggingface tokenizer and pretrained tokenizer, which uses different notation for ids/inpu_ids
        
        if pretrained_tok is None:    
            self.build_corpus()
                
            self.tokenizer = KPMTokernizer()      
            self.tokenizer.train([corpus], "./my_pretrained_bert_tok.tkn")
        else:
            using_batch_encoding = True
            self.tokenizer = KPMTokernizer(tokenizer=tokenizer,pretrained=pretrained_tok)
            # self.tokenizer.train([corpus], "pretrained_tok.tkn")

        if self.counter > 0:
            self.counter = 0
        
        labels = self.process_df(self.label_df, 'l')
        arguments = self.process_df(self.arguments_df, 'a')
        keyPoints = self.process_df(self.key_points_df, 'k')

        document_pos = []
        stances = []
        to_tensor = []
        
        
        if using_sq_classifier:
            
            for label in tqdm(labels, desc="Tokenizing "):
                document_pos.append((arguments[str(label.argId)].tfidf_pos, keyPoints[str(label.keyPointId)].tfidf_pos))
                stances.append((arguments[str(label.argId)].stance, keyPoints[str(label.keyPointId)].stance))
                label.tokenized = self.tokenizer.encode(arguments[str(label.argId)].argument, keyPoints[str(label.keyPointId)].key_point,padding='max_length', max_length=256)
                
            y = [label.label for label in labels]
            if using_batch_encoding:
                for label in tqdm(labels, desc="Preparing data"):
                    to_tensor.append(   [np.asarray(label.tokenized.input_ids, dtype=np.int32), 
                                        np.asarray(label.tokenized.attention_mask, dtype=np.int32)])
            else:
                for label in tqdm(labels, desc="Preparing data"):
                    to_tensor.append(   [np.asarray(label.tokenized.ids, dtype=np.int32), 
                                        np.asarray(label.tokenized.attention_mask, dtype=np.int32)])
        else:
            #aml = 0
            for label in tqdm(labels, desc="Tokenizing "):
                document_pos.append((arguments[str(label.argId)].tfidf_pos, keyPoints[str(label.keyPointId)].tfidf_pos))
                stances.append((arguments[str(label.argId)].stance, keyPoints[str(label.keyPointId)].stance))
                label.tokenized = [self.tokenizer.encode(arguments[str(label.argId)].argument, padding='max_length', max_length=64)]                
                label.tokenized.append(self.tokenizer.encode(keyPoints[str(label.keyPointId)].key_point, padding='max_length', max_length=64))

            ''' Utilizzato per vedere manualmente quale è la grandezza massima della sequenza per il padding da togliere
                if len(label.tokenized[-1].input_ids) > aml:
                    aml = len(label.tokenized[-1].input_ids)
                label.tokenized.append(self.tokenizer.encode(keyPoints[str(label.keyPointId)].key_point, padding='max_length', max_length=64))
                # print(f"label.tokenized: {label.tokenized}")
                if len(label.tokenized[-1].input_ids) > aml:
                    aml = len(label.tokenized[-1].input_ids)
            print("max_lenght = ", aml)
            input()
            '''

            y = [label.label for label in labels]
            if using_batch_encoding:
                for label in tqdm(labels, desc="Preparing data"):
                    to_tensor.append(   [[np.asarray(label.tokenized[0].input_ids, dtype=np.int32), 
                                        np.asarray(label.tokenized[0].attention_mask, dtype=np.int32)],
                                        [ np.asarray(label.tokenized[1].input_ids, dtype=np.int32), 
                                        np.asarray(label.tokenized[1].attention_mask, dtype=np.int32)]])
            else:
                for label in tqdm(labels, desc="Preparing data"):
                    to_tensor.append(   [[np.asarray(label.tokenized[0].ids, dtype=np.int32), 
                                        np.asarray(label.tokenized[0].attention_mask, dtype=np.int32)],
                                        [ np.asarray(label.tokenized[1].ids, dtype=np.int32), 
                                        np.asarray(label.tokenized[1].attention_mask, dtype=np.int32)]])
        return (to_tensor, y, document_pos, stances)


    def overlapping_score(self, overwrite = False, path = ""):
        
        if overwrite == True or (os.path.exists(path+"overlap_scores.txt") == False and overwrite == False): #os.path.exists(path+"overlap_scores.txt") == False:
            
            scores = []
            
            labels = self.process_df(self.label_df, 'l')
            arguments = self.process_df(self.arguments_df, 'a')
            keyPoints = self.process_df(self.key_points_df, 'k')
            
            arg_dic = {}
            kp_dic = {}

            overlapper = overlap_score(False, True, False)
    
            #arg_dic = dict(map(lambda arg: (arg[1].argId, overlapper.preprocess_text(arg[1].argument)), tqdm(arguments.items())))
            print("preprocessing arguments: ")
            results = []
            with Pool() as pool:
            # Use poolmap to apply the function to each argument in parallel
                for result in tqdm(pool.imap_unordered(overlapper.preprocess_arg, arguments.items()), total=len(arguments.items())):
                    results.append(result)
                # Close the pool
                pool.close()
                pool.join()
        
            arg_dic = dict(results)
            print("preprocessing keypoint: ")
            results = []
            with Pool() as pool:
            # Use poolmap to apply the function to each argument in parallel
                for result in tqdm(pool.imap_unordered(overlapper.preprocess_kp, keyPoints.items()), total=len(keyPoints.items())):
                    results.append(result)
                # Close the pool
                pool.close()
                pool.join()

            kp_dic = dict(results)

            scores = []
            for label in tqdm(labels):
                kp_id = label.keyPointId
                arg_id = label.argId
                
                arg_text = arg_dic[arg_id]
                kp_text = kp_dic[kp_id] 

                scores.append(round(overlapper.compute_overlap_score(arg_text,kp_text),3))

            file = open("overlap_scores.txt", "w")
            [file.write((str(val)+"\n")) for val in scores]
            file.close()
            
            return scores  
        
        elif overwrite == False:
            with open("overlap_scores.txt") as f:
                data = f.readlines()

            scores = [float(i.strip()) for i in data]
            return scores 
        
        