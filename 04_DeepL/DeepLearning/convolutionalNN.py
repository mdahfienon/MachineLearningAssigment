# Convolutional Neural Network step by step

# Question A4.3


# %% Import necessary module

import numpy as np
# import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras import layers
# from keras.optimizers import Adam
from keras.initializers import RandomNormal
# from tqdm import tqdm


# %% Loading the data

x_train = np.load("hiragana_train_imgs.npz")["arr_0"]
y_train = np.load("hiragana_train_labels.npz")["arr_0"]
x_test = np.load("hiragana_test_imgs.npz")["arr_0"]
y_test = np.load("hiragana_test_labels.npz")["arr_0"]

# %% reshaping to specify the third dimension : channels

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# %% Tranforming the output into categorical

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# %% CNN model

kernel_size = (2,2)

pooling_size = (2,2)

model = Sequential()

# add the layers
# input block
model.add(layers.Input((28, 28, 1)))


# first CNN blocks
model.add(layers.Conv2D(16, kernel_size, activation = "relu", padding="same"))
model.add(layers.MaxPooling2D(pooling_size))

# second CNN blocks
model.add(layers.Conv2D(32, kernel_size, activation = "relu", padding="same"))
model.add(layers.MaxPooling2D(pooling_size))

# second CNN blocks
model.add(layers.Conv2D(64, kernel_size, activation = "relu", padding="same"))
model.add(layers.MaxPooling2D(pooling_size))


# layers for classification

model.add(Flatten())

model.add(Dense(100, activation="relu", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))
model.add(Dense(10, activation="softmax", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))

# compile model

model.compile(optimizer=keras.optimizers.Adam(lr=0.00001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

#%% 
# sparse 0.925

x_train[0].shape
# %% model output shape

model.output_shape

#%% model summary

model.summary()

# %% fiiting the model

feedback = model.fit(x_train, y_train, 
          batch_size=20, epochs=120, use_multiprocessing=True, 
          validation_data=(x_test, y_test))

# %% feedback

feedback.history["val_accuracy"][-1]


# %% refitting 

model.fit(x_train, y_train, 
          batch_size=35, epochs=100, use_multiprocessing=True, 
          validation_data=(x_test, y_test))

# %% saving the model
model.save("cnn.model", save_format='h5')


