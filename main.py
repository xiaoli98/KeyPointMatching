import src.model as model
import tensorflow as tf
import src.dataPreprocess as dataPreprocess
import os

def main():
    data = dataPreprocess.Data()
    data.get_data_from(path="kpm_data", subset="train")
    print(data)
    siamese_model = model.Siamese_Model()
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001))
    siamese_model.fit(data, epochs=1)
    
    
if __name__== "__main__":
    main()