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
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-5,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss = tf.keras.losses.BinaryCrossentropy
model.compile(optimizer=optimizer, loss=loss, metrics=loss)

epochs = 1
total_loss = 0

print("start training")

def get_batch(train_set, index):
    batch = {}
    batch['input_ids'] = train_set['input_ids'][index]
    batch['token_type_ids'] = train_set['token_type_ids'][index]
    batch['attention_mask'] = train_set['attention_mask'][index]
    batch['labels'] = train_set['labels'][index]
    return batch
    
logits = []
for e in range(epochs):#for each epoch
    print("="*20 + "EPOCH: " + str(e) + "="*20)
    with tqdm(total=train['batches']) as pbar:
        for step in range(train['batches']):#for each batch
            logits.append(model(**get_batch(train, step)).logits)
            pbar.update(1)
        # with tf.GradientTape() as tape:
        #     output = model(**t, training=True)
        #     print(output)
            
        # grads = tape.gradient(output["loss"], model.trainable_weights) 
        # optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        # if step % 10 == 0:
        #       print("Training loss at step %d: %f" %(step, output["loss"]))
