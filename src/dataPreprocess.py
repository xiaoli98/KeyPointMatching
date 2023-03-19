from transformers import BertTokenizer
from track_1_kp_matching import *

import os
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from tokenizer.tokenizerLF import *
from tokenizer.tokenizerLF import tokenize_LF

from tokenizer.bert_tokenizer import KPMTokernizer

import random

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
    



    def make_siamese_input(self, path="kpm_data", subset="train", n_combinaitons = 3, repetition = True, pretrained_tok = None):
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
        corpus = "./src/corpuses/my_corpus"
        if not os.path.exists(corpus):      
            self.build_corpus(outFile=corpus)
        else:
            print(f"using existing corpus {corpus}")
        
        
        tokenizer = KPMTokernizer(pretrained="bert-base-cased")
        if pretrained_tok is None:          
            tokenizer.train([corpus], "./my_pretrained_bert_tok.tkn")
        else:
            tokenizer = KPMTokernizer(pretrained=pretrained_tok)


        #recover the data from the file 
        arguments_df, key_points_df, labels_file_df = self.readCSV(path, subset)

        labels = self.process_df(labels_file_df, 'l')
        arguments = self.process_df(arguments_df, 'a')
        keyPoints = self.process_df(key_points_df, 'k')
        
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
            label.tokenized = tokenizer.encode(arguments[str(label.argId)].argument+" "+arguments[str(label.argId)].topic, keyPoints[str(label.keyPointId)].key_point+" "+keyPoints[str(label.keyPointId)].topic)
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
            anchor = [arguments[str(label.argId)], keyPoints[kp_id], label.tokenized, label.label]
                      
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
                        negative = [arguments[str(fromPositive.argId)],keyPoints[str(fromPositive.keyPointId)], fromPositive.tokenized, fromPositive.label]
                        positive = [arguments[str(fromNegative.argId)],keyPoints[str(fromNegative.keyPointId)], fromNegative.tokenized, fromNegative.label]
                    else:
                        positive = [arguments[str(fromPositive.argId)],keyPoints[str(fromPositive.keyPointId)], fromPositive.tokenized, fromPositive.label]
                        negative = [arguments[str(fromNegative.argId)],keyPoints[str(fromNegative.keyPointId)], fromNegative.tokenized, fromNegative.label]
                        
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
                        negative = [arguments[str(fromPositive.argId)],keyPoints[str(fromPositive.keyPointId)], fromPositive.tokenized, fromPositive.label]
                        positive = [arguments[str(fromNegative.argId)],keyPoints[str(fromNegative.keyPointId)], fromNegative.tokenized, fromNegative.label]
                    else:
                        positive = [arguments[str(fromPositive.argId)],keyPoints[str(fromPositive.keyPointId)], fromPositive.tokenized, fromPositive.label]
                        negative = [arguments[str(fromNegative.argId)],keyPoints[str(fromNegative.keyPointId)], fromNegative.tokenized, fromNegative.label]
                        
                    #print(positive[0].printAll(), positive[1].printAll())
                    #print(negative[0].printAll(), negative[1].printAll())
                    
                    an_pos_neg.append([anchor, positive, negative])                    
        return an_pos_neg

    def test_make_siamese_input(self, path="kpm_data", subset="train", n_combinaitons = 3, repetition = True, pretrained_tok = None):
        d = Data()
        asd = d.make_siamese_input(path = path, subset = subset, n_combinaitons=n_combinaitons, repetition = repetition, pretrained_tok=pretrained_tok)
        #print(asd[:5])
        print(type(asd))
        
        good_input = 0
        wrong_input = 0
            
        for i in tqdm(range(0,len(asd))):
            
            if (asd[i][1][3] == asd[i][0][3] and asd[i][2][3] != asd[i][0][3] ):
                good_input+=1
            else:
                print(asd[i][0][0].printAll())
                print(asd[i][0][1].printAll())
                print(asd[i][0][2])
                print(asd[i][0][3])
                print("+++++")
                print(asd[i][1][0].printAll())
                print(asd[i][1][1].printAll())
                print(asd[i][1][2])
                print(asd[i][1][3])
                print("+++++")
                print(asd[i][2][0].printAll())
                print(asd[i][2][1].printAll())
                print(asd[i][2][2])
                print(asd[i][2][3])
                print("---")
                
                wrong_input+=1
        
        print("the correct inputs are: ", good_input, " out of ", len(asd))
        print("the wrong inputs are: ", wrong_input, " out of ", len(asd))
        error_rate = (wrong_input/len(asd))*100
        print("error rate is: ",error_rate,"%")
        
        correct_rate = (good_input/len(asd))*100
        print("correct rate is: ",correct_rate,"%")

        return asd
    
    def get_tf_dataset(self, path="kpm_data", subset="train", n_combinaitons = 3, repetition = True, pretrained_tok = None, test_input = False):
        
        my_list = None

        if test_input:
            my_list = self.test_make_siamese_input(path = path, subset = subset, n_combinaitons=n_combinaitons, repetition = repetition, pretrained_tok=pretrained_tok)
        else:
            my_list = self.make_siamese_input(path = path, subset = subset, n_combinaitons=n_combinaitons, repetition = repetition, pretrained_tok=pretrained_tok)
        #np.dtype(list)
        """
        print(my_list[0][0][2])
        print(my_list[0][0][2].ids)
        print(type(my_list[0][0][2].ids))
        print()
        print(my_list[0][0][2].type_ids)
        print(type(my_list[0][0][2].type_ids))
        print()
        print(my_list[0][0][2].tokens)
        print(type(my_list[0][0][2].tokens))
        print()
        print(my_list[0][0][2].offsets)
        print(type(my_list[0][0][2].offsets))
        print()
        print(my_list[0][0][2].attention_mask)
        print(type(my_list[0][0][2].attention_mask))
        print()
        print(my_list[0][0][2].special_tokens_mask)
        print(type(my_list[0][0][2].special_tokens_mask))
        print()
        print(my_list[0][0][2].overflowing)
        print(type(my_list[0][0][2].overflowing))
        print()
        print(type(my_list[0][0][2]))
        input()
        
        dataset = tf.data.Dataset.from_generator( lambda: ((x[0][2][1], x[1][2][1], x[2][2][1]) for x in my_list),
            output_types=(np.dtype(list) , np.dtype(list), np.dtype(list)))
        print(len(dataset))
        return dataset.map(lambda ancor, pos, neg: self.parse_fn(ancor, pos, neg))
        """
        to_transform = []
        for i in tqdm(range (0,len(my_list))):
            anchor = [my_list[i][0][2].ids, my_list[i][0][2].type_ids, my_list[i][0][2].offsets, my_list[i][0][2].attention_mask, my_list[i][0][2].special_tokens_mask, my_list[i][0][2].overflowing]
            positive = [my_list[i][1][2].ids, my_list[i][1][2].type_ids, my_list[i][1][2].offsets, my_list[i][1][2].attention_mask, my_list[i][1][2].special_tokens_mask, my_list[i][1][2].overflowing]
            negative = [my_list[i][2][2].ids, my_list[i][2][2].type_ids, my_list[i][2][2].offsets, my_list[i][2][2].attention_mask, my_list[i][2][2].special_tokens_mask, my_list[i][2][2].overflowing]
            to_transform.append([anchor, positive, negative]) 
        
        return tf.data.Dataset.from_tensor_slices(to_transform)
        
    # Define parsing function to extract features
    def parse_fn(self, obj1, obj2, obj3):
        return {'ancor': obj1,
                'pos': obj2,
                'neg': obj3}

d = Data()
tw_ds = d.get_tf_dataset(n_combinaitons = 1, repetition = True)
print(type(tw_ds))
##print(len(tw_ds))

for example in tw_ds:
    print(example)
    input()
