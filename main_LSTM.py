import datetime
import numpy as np
import tensorflow as tf
import src.dataPreprocess as dataPreprocess
import src.LSTMModel as LSTMModel

import simple_tokenizer as tokenize

from tensorflow.python.framework.config import set_memory_growth
def useGPU():
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.set_visible_devices(physical_devices[1], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[1], True)
        tf.config.set_visible_devices(physical_devices[2], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[2], True)
    except:
     #Invalid device or cannot modify virtual devices once initialized.
        pass


def main():
    useGPU()
    data = dataPreprocess.Data()
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime(f"%m%d-%H%M-LSTM")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    X_train, y_train, vocab_size = tokenize.tokenize_LSTM()
    input_length = len(X_train[0])
    X_dev, y_dev, _ = tokenize.tokenize_LSTM(path="kpm_data", subset="dev", pad_to_length=input_length)

    BATCH_SIZE = 64
    EPOCHS = 150
    input_size = list(np.shape(X_train))

    lstm = LSTMModel.LSTMModel(vocab_size=vocab_size, max_length = input_size[1], name="LSTM")
    opt = tf.keras.optimizers.Adam(2e-5)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    lstm.compile(optimizer=opt,
                loss= loss_fn,
                metrics=[
                        tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall()
                        ]
                        )
    
    print("start training")
    lstm.fit(x=X_train, 
                y=np.array(y_train), 
                validation_data=(X_dev, np.array(y_dev)),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                callbacks=[tensorboard_callback],
                verbose=1)
    lstm.save(f"models/{datetime.datetime.now().strftime(f'%m%d-%H%M')}-LSTM")
    
    data_dev = dataPreprocess.Data(subset="dev")
    
    out_dev = lstm.predict(X_dev)
    with open(f"./prediction_dev_LSTM.txt", "w") as f:
        for i in range(len(out_dev)):
            f.write(f"{out_dev[i]}\n")

if __name__== "__main__":
    main()