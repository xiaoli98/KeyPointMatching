import numpy as np
import tensorflow as tf

from tensorflow import keras

class DistanceLayer():
    def __init__(self, doc_feat_matrix, metric) -> None:
        super().__init__()
        
        self.doc_feat_matrix = doc_feat_matrix
        
        if metric == "cosine":
            self.distance = self.cosine_sim
        elif metric == "jaccard":
            self.distance = self.jaccard
        else:
            raise ValueError("distance metric not found")
    
    def jaccard(self, a, b)->float:
        """compute jaccard similarity

        Args:
            a: first array
            b: second array

        Returns:
            float: jaccard similarity between a and b
        """
        features_a = a.nonzero()[1]
        features_b = b.nonzero()[1]
        c = np.intersect1d(features_a, features_b)
        return float(len(c)) / (len(features_a) + len(features_b) - len(c))

    def cosine_sim(self, a, b)->float:
        """compute cosine similarity

        Args:
            a: first array
            b: second array

        Returns:
            float: cosine similarity between a and b
        """
        
        dot = a.dot(b.transpose())
        
        magnitude_a = np.sqrt(a.power(2).sum())
        magnitude_b = np.sqrt(b.power(2).sum())
        
        return np.divide(dot.data, magnitude_a*magnitude_b)
        
    def compute(self, X):
        doc1 = self.doc_feat_matrix[X[0]]
        doc2 = self.doc_feat_matrix[X[1]]
        return self.distance(doc1, doc2)
