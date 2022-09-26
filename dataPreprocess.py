from track_1_kp_matching import *

#TODO
#la tokenizzazione si fa dopo aver creato il dataset
#il dataset e' composto da fare una join tra arg_id e key_id presente nel label_file
#

#implementare modelli tipo distillated BERT/BERT_based 


class Argument:
    @property
    def argId(self):
        return self.argId
    
    @property
    def argument(self):
        return self.argument
    
    @property
    def topic(self):
        return self.topic
    
    @property
    def stance(self):
        return self.stance
    
    @argId.setter
    def argId(self, argId):
        self.argId = argId
        
    @argument.setter
    def argument(self, argument):
        self.argument = argument
        
    @topic.setter
    def topic(self, topic):
        self.topic = topic
        
    @stance.setter
    def stance(self, stance):
        self.stance = stance
        

class KeyPoint:
    @property
    def KeyPointId(self):
        return self.KeyPointId
    
    @property
    def key_point(self):
        return self.key_point
    
    @property
    def topic(self):
        return self.topic
    
    @property
    def stance(self):
        return self.stance
    
    @KeyPointId.setter
    def KeyPointId(self, KeyPointId):
        self.KeyPointId = KeyPointId
        
    @key_point.setter
    def key_point(self, key_point):
        self.key_point = key_point
        
    @topic.setter
    def topic(self, topic):
        self.topic = topic
        
    @stance.setter
    def stance(self, stance):
        self.stance = stance
        

class Label:
    @property
    def KeyPointId(self):
        return self.KeyPointId
    
    @property
    def argId(self):
        return self.argId
    
    @property
    def label(self):
        return self.label
    
    
    @KeyPointId.setter
    def KeyPointId(self, KeyPointId):
        self.KeyPointId = KeyPointId
        
    @argId.setter
    def argId(self, argId):
        self.argId = argId
        
    @label.setter
    def label(self, label):
        self.label = label
        

def preprocess(path="kpm_data", subset="train"):
    arguments_df, key_points_df, labels_file_df = load_kpm_data(path, subset)
    
    

