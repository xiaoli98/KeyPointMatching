import tensorflow as tf

from src.dataPreprocess import *
from src.Siamese import SiameseBert

class Siamese_Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.siameseNet = SiameseBert()
        
        # todo
        self.margin = 0.5
        
    def call(self, data):
        anchor = data[0]
        positive = data[1]
        negative = data[2]
        return self.siameseNet(anchor, positive, negative)
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siameseNet.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.siameseNet.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def test_step(self, data):
        loss = self._compute_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def _compute_loss(self, data):
        anchor = data[0]
        positive = data[1]
        negative = data[2]
        ap_distance, an_distance = self.siameseNet(anchor, positive, negative)
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss