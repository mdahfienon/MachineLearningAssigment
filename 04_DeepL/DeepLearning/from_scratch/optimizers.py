#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux 
@date : 2020 Jul 09
@last modified : 2021 Apr 06, 17:52:27
"""

import numpy as np

class Optimizer:
    def update(self, *args, **kwargs):
        abstract


class StochasticGradientDescent(Optimizer):
    def __init__(self, lr, momentum=0):
        self.lr = lr
        self.momentum = momentum
        self.cur_w = None

    def update(self, w, grad_w):
        if self.cur_w is None:
            self.cur_w = np.zeros_like(w)

        self.cur_w = self.momentum * self.cur_w + (1 - self.momentum) * grad_w
        return w - self.lr * self.cur_w

class GradientDescent(StochasticGradientDescent):
    def __init__(self, lr):
        super().__init__(lr, momentum=0)
