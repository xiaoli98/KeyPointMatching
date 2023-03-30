import tensorflow as tf

from tensorflow import keras

class DistanceLayer(keras.layers.Layer):
    def __init__(self, metric) -> None:
        super().__init__()
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
        a = set(a.split()) 
        b = set(b.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    def cosine_sim(self, a, b)->float:
        """compute cosine similarity

        Args:
            a: first array
            b: second array

        Returns:
            float: cosine similarity between a and b
        """
        
        dot = tf.tensordot(a, b, axes=0)
        magnitude_a = tf.norm(a)
        magnitude_b = tf.norm(b)
        
        return tf.math.divide(dot, tf.multiply(magnitude_a, magnitude_b))
        
    def call(self, X1, X2, metric="cosine", vectorizer=None):
        return self.distance(X1, X2)
