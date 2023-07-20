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
    
    X_train, y_train = tokenize.tokenize_LSTM()

    batch_size = 16
    input_size = list(np.shape(X_train))

    lstm = LSTMModel.LSTMModel(max_length = input_size[1], name="LSTM")
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
                epochs=5,
                batch_size=batch_size,
                shuffle=True,
                callbacks=[tensorboard_callback],
                verbose=1)
    lstm.save(f"models/{pretrained}-{hs}")

if __name__== "__main__":
    main()