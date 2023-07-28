import datetime
import numpy as np
import tensorflow as tf
import src.dataPreprocess as dataPreprocess
import src.Conv1D as Conv1D
import os
import simple_tokenizer as tokenize


def main():
    data = dataPreprocess.Data()
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime(f"%m%d-%H%M-conv1d")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    X_train, y_train, vocab_size = tokenize.tokenize_LSTM()
    input_length = len(X_train[0])
    X_dev, y_dev, _ = tokenize.tokenize_LSTM(path="kpm_data", subset="dev", pad_to_length=input_length)

    BATCH_SIZE = 6
    EPOCHS = 180
    input_size = list(np.shape(X_train))

    conv = Conv1D.Convolution(vocab_size=vocab_size, max_length = input_size[1], name="conv1d")
    opt = tf.keras.optimizers.Adam(2e-5)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    conv.model.compile(optimizer=opt,
                loss= loss_fn,
                metrics=[
                        tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall()
                        ]
                )
    conv.model.summary()
    print("--sizes--")
    print("X_train ", np.shape(X_train))
    print("y_train ", np.shape(y_train))
    print("X_dev ", np.shape(X_dev))
    print("y_dev ", np.shape(y_dev))
    print("start training")
    conv.model.fit(x=X_train, 
            y=np.array(y_train), 
            validation_data=(X_dev, np.array(y_dev)),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            callbacks=[tensorboard_callback],
            verbose=1)
    conv.save(f"models/Conv1d-{EPOCHS}-{BATCH_SIZE}")

    data_dev = dataPreprocess.Data(subset="dev")
    
    out_dev = conv.predict(X_dev)
    with open(f"./prediction_dev_Conv1d.txt", "w") as f:
        for i in range(len(out_dev)):
            f.write(f"{out_dev[i]}\n")

if __name__== "__main__":
    main()