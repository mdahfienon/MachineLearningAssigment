# DeepLearning step by step

# Question A4.2


# %% Import necessary module

import numpy as np
# import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from tqdm import tqdm


# %% Loading the data

x_train = np.load("hiragana_train_imgs.npz")["arr_0"]
y_train = np.load("hiragana_train_labels.npz")["arr_0"]
x_test = np.load("hiragana_test_imgs.npz")["arr_0"]
y_test = np.load("hiragana_test_labels.npz")["arr_0"]


# %% Tranforming the output into categorical

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


# %% Checking the shape of the data

print(x_train.shape, y_train.shape)

print(y_train[0])

x_train[0][0]


# %% First model through sequential

model = Sequential()

# adding the flatten layer

model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(10, activation="softmax", 
                kernel_initializer=RandomNormal(), 
                bias_initializer=RandomNormal()))

# compile model

rate = 0.00001
optimizer = Adam(learning_rate=rate)

model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# %% Checking result output

model.output_shape



# %% Fitting first model

model.fit(x_train, y_train, 
          batch_size=32, epochs=100, use_multiprocessing=True, 
          validation_data=(x_test, y_test), verbose=0)

# %% saving first model

model.save("firstlinear.model", save_format='h5')

# %% summary of first model

model.summary()

# %% tanh model architecture

# non-linear NN
modeltanh = Sequential()

# adding the flatten layer

modeltanh.add(Flatten(input_shape=(28, 28)))
modeltanh.add(Dense(100, activation="tanh", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))
modeltanh.add(Dense(10, activation="softmax", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))

# compile model

rate = 0.00001
optimizer = Adam(learning_rate=rate)

modeltanh.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])


# %% tanh summary
modeltanh.summary()

# %% fitting tanh model and saving it

modeltanh.fit(x_train, y_train, 
              batch_size=32, epochs=100, use_multiprocessing=True, 
              validation_data=(x_test, y_test), verbose=0)

modeltanh.save("nonlineartanh.model", save_format='h5')



# %% relu model architecture

modelRelu = Sequential()

# adding the flatten layer

modelRelu.add(Flatten(input_shape=(28, 28)))
modelRelu.add(Dense(100, activation="relu", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))
modelRelu.add(Dense(10, activation="softmax", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))

# compile model

rate = 0.00001
optimizer = Adam(learning_rate=rate)

modelRelu.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

#%% Fitting of relu
modelRelu.fit(x_train, y_train, 
              batch_size=32, epochs=100, use_multiprocessing=True, 
              validation_data=(x_test, y_test))

modelRelu.save("Relulinear.model", save_format='h5')

# %% Which activation is better




# %% Running 10 tanh

# 10 run of modeltanh
tanhTestAcc = []
for i in range(10):
    tanhhistory = modeltanh.fit(x_train, y_train, 
              batch_size=32, epochs=100, use_multiprocessing=True, 
              validation_data=(x_test, y_test), verbose=0)
    
    tanhTestAcc.append(tanhhistory.history['val_accuracy'][99])

tanhTestAcc


# %% Running 10 relu 

# 10 run of relu
reluTestAcc = []
for i in range(10):
    reluhistory = modelRelu.fit(x_train, y_train, 
              batch_size=32, epochs=100, use_multiprocessing=True, 
              validation_data=(x_test, y_test), verbose=0)
    
    reluTestAcc.append(reluhistory.history['val_accuracy'][99])

reluTestAcc


# %% Trying multiprocessing for relu
from multiprocessing import Pool
multiRelu=[]
with Pool(processes=6) as pool:
    for i in pool.imap_unordered(modelRelu.fit(x_train, y_train, 
              batch_size=32, epochs=100, use_multiprocessing=True, verbose=0,
              validation_data=(x_test, y_test)).history['val_accuracy'][99], 
                                 range(10)):
        multiRelu.append(i)
        
    
multiRelu
# %% get mean of multiRelu

np.mean(multiRelu)


# %%  Trying multiprocessing for tanh

multitanh = []
with Pool(processes=6) as pool:
    for i in pool.imap_unordered(modeltanh.fit(x_train, y_train, 
              batch_size=32, epochs=100, use_multiprocessing=True, 
              validation_data=(x_test, y_test), verbose=0).history['val_accuracy'][99], 
                                 range(10)):
        multitanh.append(i)
        
    
multitanh

# %% get mean of multitanh

np.mean(multitanh)


# %% Parallelization of the running

from threading import Thread
# from time import time
# from tqdm import tqdm


def get_accuracies(activ_func, output, hidden_size = 100):
    
    """
    activ_func : activation function
    
    output : a list to append the accuracy
    
    """
    
    def get_result():
        """
        the model architecture
        """
        
        model = Sequential()

        # adding the flatten layer

        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dense(hidden_size, activation=activ_func, kernel_initializer=RandomNormal(), 
                            bias_initializer=RandomNormal()))
        model.add(Dense(10, activation="softmax", kernel_initializer=RandomNormal(), 
                            bias_initializer=RandomNormal()))

        # compile model
        optimizer = Adam(learning_rate=0.00001)

        model.compile(optimizer=optimizer, 
                          loss='categorical_crossentropy', 
                          metrics=['accuracy'])
        
        return model
    
    for j in tqdm(range(10)):
        model = get_result()
        feedback = model.fit(x_train, y_train, 
                  batch_size=32, epochs=100, use_multiprocessing=True, 
                  validation_data=(x_test, y_test), verbose=0)
        
        accu = feedback.history["val_accuracy"][-1]
        
        output.append(accu)
        
    return output
        
# %% Creating the thread

reluOutput = list()
tanhOutput = list()

relu_thread = Thread(target=get_accuracies, 
                     kwargs = dict(activ_func = "relu", output=reluOutput ))

tanh_thread = Thread(target=get_accuracies, 
                     kwargs = dict(activ_func = "tanh", output =tanhOutput ))


# %% running the thread

relu_thread.start(); tanh_thread.start();
relu_thread.join(); tanh_thread.join();
 
# %% mean accuracies for relu

np.mean(reluOutput)

 # %% mean accuracies for tanh

np.mean(tanhOutput)

# %% changing hidden layer size 

relu10Output = list()
relu100Output = list()
relu1000Output = list()

relu10_thread = Thread(target=get_accuracies, 
                     kwargs = dict(activ_func = "relu", output=relu10Output,
                                   hidden_size = 10))

relu100_thread = Thread(target=get_accuracies, 
                     kwargs = dict(activ_func = "relu", 
                                   output=relu100Output))


relu1000_thread = Thread(target=get_accuracies, 
                     kwargs = dict(activ_func = "relu", 
                                   output=relu1000Output,
                                   hidden_size = 1000))

# %% running the changed layers architecture

relu10_thread.start(); relu100_thread.start(); relu1000_thread.start()
relu10_thread.join(); relu100_thread.join(); relu1000_thread.join()

# %%

print(np.mean(relu10Output),
      np.mean(relu100Output),
      np.mean(relu1000Output), end="\n")

# %% results of the running

print(f'relu10_accu = {np.mean(relu10Output)} \nrelu100_accu = {np.mean(relu100Output)} \nrelu1000_accu = {np.mean(relu1000Output)}')

# %% model with 3 layers of 1000 units

# unit = list([10, 100, 1000])

reluStudy = Sequential()

# adding the flatten layer

reluStudy.add(Flatten(input_shape=(28, 28)))
reluStudy.add(Dense(1000, activation="relu", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))
reluStudy.add(Dense(1000, activation="relu", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))
reluStudy.add(Dense(1000, activation="relu", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))
reluStudy.add(Dense(10, activation="softmax", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))

# compile model
optimizer = Adam(learning_rate=0.00001)

reluStudy.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

feed = reluStudy.fit(x_train, y_train, 
              batch_size=32, epochs=100, use_multiprocessing=True, 
              validation_data=(x_test, y_test), verbose=0)

feed.history["val_accuracy"]

# test accuracy here is 0.8718

#%% relu1Study model


relu1Study = Sequential()

# adding the flatten layer

relu1Study.add(Flatten(input_shape=(28, 28)))
relu1Study.add(Dense(1000, activation="relu", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))
relu1Study.add(Dense(1000, activation="relu", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))
relu1Study.add(Dense(100, activation="relu", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))
relu1Study.add(Dense(10, activation="softmax", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))

# compile model
optimizer = Adam(learning_rate=0.00001)

relu1Study.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

feed1 = relu1Study.fit(x_train, y_train, 
              batch_size=32, epochs=100, use_multiprocessing=True, 
              validation_data=(x_test, y_test), verbose=0)

feed1.history["val_accuracy"][-1]

#%% relu2Study model 

relu2Study = Sequential()

# adding the flatten layer

relu2Study.add(Flatten(input_shape=(28, 28)))
relu2Study.add(Dense(1000, activation="relu", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))
relu2Study.add(Dense(1000, activation="relu", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))
relu2Study.add(Dense(10, activation="relu", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))
relu2Study.add(Dense(10, activation="softmax", kernel_initializer=RandomNormal(), 
                    bias_initializer=RandomNormal()))

# compile model
optimizer = Adam(learning_rate=0.00001)

relu2Study.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

feed2 = relu2Study.fit(x_train, y_train, 
              batch_size=32, epochs=100, use_multiprocessing=True, 
              validation_data=(x_test, y_test), verbose=0)

feed2.history["val_accuracy"][-1]

# %% 

