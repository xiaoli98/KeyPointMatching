import datetime
import numpy as np
import tensorflow as tf
import src.dataPreprocess as dataPreprocess
import src.LSTMModel as LSTMModel

def main():
    data = dataPreprocess.Data()
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime(f"%m%d-%H%M-LSTM")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    lstm = LSTMModel(name="LSTM")
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
                batch_size=16,
                shuffle=True,
                callbacks=[tensorboard_callback],
                verbose=1)
    lstm.save(f"models/{pretrained}-{hs}")

if __name__== "__main__":
    main()