import tensorflow as tf
import pandas as pd

from dataPreprocess import *
from transformers import TFBertForSequenceClassification as bert



train = preprocess()

#for d in data:
#    train.append(d[2])
#    label.append(d[3])

#print(len(train))
#print(len(label))



model = bert.from_pretrained("ydshieh/bert-base-uncased-yelp-polarity")
optimizer = tf.keras.optimizers.Adam(3e-5)
loss = tf.keras.losses.BinaryCrossentropy
model.compile(optimizer=optimizer, loss=loss)

epochs = 1
total_loss = 0

print("start training")

for e in range(epochs):#for each epoch
    print("="*20 + "EPOCH: " + str(e) + "="*20)
    for step, t in enumerate(train):#for each batch
        with tf.GradientTape() as tape:
            output = model(**t, training=True)
            print(output)
            
        grads = tape.gradient(output["loss"], model.trainable_weights) 
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        if step % 10 == 0:
              print("Training loss at step %d: %f" %(step, output["loss"]))
