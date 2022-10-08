import tensorflow as tf
import pandas as pd

from dataPreprocess import *
from transformers import TFBertForSequenceClassification as bert



data = Data()
data.get_data_from(path="kpm_data", subset="train")

model = bert.from_pretrained("ydshieh/bert-base-uncased-yelp-polarity")
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-5,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)

epochs = 1
batch = 16

print("start training")

model.fit(data.shuffle(1000).batch(batch), epochs=epochs, batch_size=batch, verbose=1)
# logits = []
# for e in range(epochs):#for each epoch
#     print("="*20 + "EPOCH: " + str(e) + "="*20)
#     with tqdm(total=train['batches']) as pbar:
#         for step in range(train['batches']):#for each batch
#             batch, labels = get_batch(train, step)
#             model.fit(batch, labels)
#             # logits.append(model(**get_batch(train, step)).logits)
#             pbar.update(1)
#         # with tf.GradientTape() as tape:
#         #     output = model(**t, training=True)
#         #     print(output)
            
#         # grads = tape.gradient(output["loss"], model.trainable_weights) 
#         # optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
#         # if step % 10 == 0:
#         #       print("Training loss at step %d: %f" %(step, output["loss"]))
