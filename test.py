import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

width = 10
stride = 5
number_of_frames = 2
batch_size = 100
img_size = 100
lr = 1e-3


if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)

    model = keras.models.Sequential()
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu'),input_shape=(b,img_size, img_size,1)))
    model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(3,3))))
    model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(20,return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(20)))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(6)

    # load weights
    model.load_weights("weights.best.hdf5")

    with open('\X.pickle', 'rb') as f:
        X = pickle.load(f)
    with open('\y.pickle', 'rb') as f:
        y = pickle.load(f)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    # estimate accuracy on whole dataset using loaded weights
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))