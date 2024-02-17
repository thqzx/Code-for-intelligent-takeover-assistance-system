from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras


width = 10
stride = 5
number_of_frames = 2
batch_size = 100
img_size = 100
lr = 1e-3


with open('\X.pickle', 'rb') as f:
    X = pickle.load(f)
with open('\y.pickle', 'rb') as f:
    y = pickle.load(f)

    
if __name__ == '__main__':
    model = keras.models.Sequential()
    results = []
    # X,y = shuffle(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                           input_shape=(number_of_frames, img_size, img_size, 1)))
    model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(3, 3))))
    model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(20, return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(20)))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(6)
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['acc']

    # checkpoint
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    # Fit the model
    history = model.fit(X_train, y_train, batch_size=batch_size, callbacks=callbacks_list, epochs=50, verbose=1, shuffle=False)
    results.append('Input shape:%s ,frame length: %s,pattern size: %s,pattern hop:%s,Train: %.3f,Test: %.3f' % (
    X.shape, width, number_of_frames, stride, history.history.get('acc')[-1])