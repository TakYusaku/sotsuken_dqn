
#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras import backend as K
from keras.utils import np_utils

def loadData(filename):
    data = np.loadtxt(filename, delimiter=',')

def convertCNNInput(X, t, img_rows=28, img_cols=28, channel=1):
    # 入力を28x28の2次元行列に(元は784の長さのベクトル)
    if K.image_data_format() == 'channels_first':
        X = X.reshape(X.shape[0], channel, img_rows, img_cols)
        input_shape = (channel, img_rows, img_cols)
    else:
        X = X.reshape(X.shape[0], img_rows, img_cols, channel)
        input_shape = (img_rows, img_cols, channel)

    one_hot_t = np_utils.to_categorical(t)

    return X,one_hot_t
