import tensorflow as tf

from dataPreprocess import *
from transformers import TFBertForSequenceClassification as bert

model = bert.from_pretrained("ydshieh/bert-base-uncased-yelp-polarity")

model.compile(optimizer=tf.keras.optimizers.Adam(3e-5),
              loss=tf.keras.losses.BinaryCrossentropy
              )

data = preprocess()

train = []
label = []

for d in data:
    train.append(d[2])
    label.append(d[3])

label = np.array(label)
#print(len(train))
#print(len(label))
model.fit(train, label)