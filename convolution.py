# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:38:24 2024

@author: Кормак
"""

import numpy as np
import matplotlib.pyplot as plt

from network import BaseConvLayer
from STBP import *
from optimizers import ADAMW

def train(eps):
    i = np.array([
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
        ])
    i1 = np.array([np.concatenate(i) for _ in range(10)]).transpose()
    i2 = np.array([np.concatenate(i.transpose()) for _ in range(10)]).transpose()
    layers = []
    layers.append(ConvLayer(    
        weights=[np.random.rand(3, 3) for _ in range(3)], 
        neurons=['n1', 'n2', 'n3'], 
        thres=3,
        inputs=[f"{i}" for i in range(i.shape[0]*i.shape[1])],
        i=i.shape,
        s=(1,1),
        p=0,
        learning=True,
        opt_class=ADAMW,
        opt_params={"a":0.2}
    )),
    layers.append(FCLayer(
        neurons = np.array(['nO.1', 'nO.2']), 
        thres = 1,
        prevl = layers[-1], 
        weights = Network_STBP.random_weights(2, len(layers[-1].neurons)),
        opt_class = ADAMW,
        opt_params = {"a":0.2}
    ))
    net = Network_STBP(layers, 1, 1)
    loss = np.inf
    k = 0
    while loss > eps or accuracy < 1:
        net.fit([i1, i2]*100, [np.array((0,1)), np.array((1,0))]*100, discrete_input=True)
        loss = net.score([i1, i2]*100, [np.array((0,1)), np.array((1,0))]*100, discrete_input=True)
        print(f"step {k}, {loss=}")
        accuracy = net.accuracy([i1, i2]*100, [np.array((0,1)), np.array((1,0))]*100, discrete_input=True)
        print(f"{accuracy=}")
        k += 1

if __name__ == "__main__":
    train(eps=0.05)