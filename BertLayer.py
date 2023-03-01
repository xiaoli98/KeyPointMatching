import tensorflow as tf

from transformers import TFBertModel as bert
from tensorflow import keras

class BertLayer(keras.layers.Layer):
    """Bert Layer, it can do also the classification of sequence if provided a well defined classifier
    
    to define the classifier, pass a tf sequential model and use add(layer) add more layers

    """
    
    def __init__(self, bert_model_name, classifier:keras.Sequential=None) -> None:
        """initialize the bert model with a pretrained one, bert_model_name should be an existining model on hugging face

        Args:
            bert_model_name (string): an existing pretrained bert model
            classifier (keras.Sequential, optional): a sequential model for sequence classification. Defaults to None.
        """
        super().__init__()
        self.model = bert.from_pretrained(bert_model_name)
        self.classifier = classifier
        
    def call(self, input_id, mask = None):
        """override of tf call

        Args:
            input_id (_type_): _description_
            mask (_type_, optional): _description_. Defaults to None.
        """
        _, pooled_output = self.model(input_ids= input_id, attention_mask=mask,return_dict=False)
        if(self.classifier is not None):
            output = self.classifier(pooled_output)
    
    def add(self, layer:keras.layers.Layer):
        """add a layer to the classifier

        Args:
            layer (keras.layers.Layer): _description_
        """
        self.classifier.add(layer)
        