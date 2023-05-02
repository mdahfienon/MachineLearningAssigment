#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2020 Jul 09
@last modified : 2021 Apr 06, 17:46:29
"""
import numpy as np


class Loss:
    def __init__(self):
        pass

    def __call__(self, y, y_hat, *args, **kwargs):
        return self.loss(y, y_hat, *args, **kwargs)

    def loss(self, y, y_hat):
        raise NotImplementedError()

    def gradient(self, y, y_hat):
        raise NotImplementedError()


class MSE(Loss):
    def loss(self, y, y_hat):
        return 0.5 * np.power(y - y_hat, 2)

    def gradient(self, y, y_hat):
        return y_hat - y


class Crossentropy(Loss):
    def loss(self, y, y_hat):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

    def gradient(self, y, y_hat):
        y_hat += 1e-15
        return -(y / y_hat) + (1 - y) / (1 - y_hat)


class SparseCrossentropy(Loss):
    def loss(self, y, p):
        y_one_hot = np.zeros_like(p)
        y_one_hot[np.arange(len(y)), y] = 1
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -y_one_hot * np.log(p) - (1 - y_one_hot) * np.log(1 - p)

    def gradient(self, y, p):
        y_one_hot = np.zeros_like(p)
        y_one_hot[np.arange(len(y)), y] = 1
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(y_one_hot / p) + (1 - y_one_hot) / (1 - p)
