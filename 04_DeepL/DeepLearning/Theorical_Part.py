# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:22:38 2023

@author: MATHIAS
"""

# %% relating my path to the from_scratch path


import os, sys

fs_path = os.path.join(os.getcwd(), "from_scratch")
if fs_path not in sys.path:
    sys.path.append(fs_path)

# %% importing needed mudules


import from_scratch as fs
import numpy as np
from from_scratch.models import Sequential
from from_scratch import layers, losses, optimizers

#%% x_or problem instantiation


X_or = np.array([[0,0], [0,1], [1,0], [1,1]])

y_or = np.array([0,1,1,0]).reshape(-1,1)


# %% 

print(X_or, "\n", y_or)

# %% weight and bias



wxh = np.array([[0.81, 1.3], [0.9, 1.2]])

bh = np.array([0.8, -1.19])

why = np.array([[0.65], [-2.0]])

by = np.array([-0.28])

# %% model architecture


model = Sequential([layers.Dense(2, activation="relu"),
                    layers.Dense(1, activation="sigmoid")])

model.compile(losses.Crossentropy(), 
              optimizer=optimizers.GradientDescent(lr=0.3))

# %% feeding forward

run = model.forward(X_or)


# %% initialization of the feedforward

model.layers[0].weights = wxh

model.layers[0].biases = bh

model.layers[2].weights = why

model.layers[2].biases = by

# %% model summary


print(model.summary())

# %% run the network

y_hat = model.forward(X_or)

# %% coorect prediction


def correctness(y_hat):
    y_hat_round = np.round(y_hat)
    return np.sum(y_or == y_hat_round)

correctness(y_hat)

# %% the loss


loss_fn = fs.losses.Crossentropy()

loss = np.sum(loss_fn(y_or, y_hat))

loss

# %% output layer gradient


xback = np.array([[0,0]])

yback = np.array([[0]])

ybackhat = model.forward(xback)

loss_grad = loss_fn.gradient(yback, ybackhat)[0,0]

loss_grad


# %% gradient of weights

sigmoid_grad = model.layers[-1].backward(loss_grad)

why_grad = model.layers[-2].backward(sigmoid_grad, get_all = True)

grad_why = why_grad["weights"]

grad_b_hy = why_grad["biases"]

print(grad_why[0,0], grad_why[1,0], grad_b_hy[0,0])

# %% gradient of hidden layers

print(
why_grad["inputs"][0,0],

why_grad["inputs"][0,1], sep='\n')


# %% first layers gradient


relu_grad = model.layers[-3].backward(why_grad["inputs"])

wxh_grad = model.layers[-4].backward(relu_grad, get_all=True)

grad_wxh = wxh_grad["weights"]
grad_bxh = wxh_grad["biases"]

print(grad_wxh[0,0], grad_wxh[0,1], grad_wxh[1,0], grad_wxh[1,1], sep="\n")


# %%

print(grad_bxh, grad_wxh, sep="\n\n\n")

# %% weights update

print(model.layers[-2].weights, model.layers[-2].biases, sep='\n\n')

# %% first layers weights update 

print(model.layers[0].weights, model.layers[0].biases, sep="\n\n")

# %% plot in the hidden layer

import matplotlib.pyplot as plt

h = model.layers[1].forward(model.layers[0].forward(X_or))
y = np.round(model.forward(X_or)).reshape(-1)

zeros = y == 0

plt.figure()
plt.ylabel(r"$h_2$")
plt.xlabel(r"$h_1$")

plt.scatter(h[zeros, 0], h[zeros, 1], color = "purple", label="0")
plt.scatter(h[~zeros, 0], h[~zeros, 1], color = "blue", label="1")
plt.legend()
plt.show()


# %% output of all point

y_hat_final = model.forward(X_or) 

y_hat_final


# %% correct classification 

correctness(y_hat_final)

# %% loss

np.sum(model.loss_function(y_or, y_hat_final))


# %% affirmations 

history = model.fit(X_or, y_or, epochs = 200, verbose = 0)

final_y = model.forward(X_or).reshape(-1)

final_y



# %% plt

plt.figure()
plt.ylabel("cross entropy")
plt.xlabel("Epochs")
plt.semilogx(history["epochs"], history["loss"])
plt.legend()
plt.show()





