#! /usr/bin/python
# -*- coding: utf-8 -*-
#import plaidml.keras
#plaidml.keras.install_backend()

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import matplotlib.pyplot as plt
import pickle
from keras.datasets import mnist
import common2
import numpy as np

class CNN:
    def __init__(self):
        NUM_CATEGORY = 10
        NUM_H1 = 40
        img_rows=28
        img_cols=28

        if K.image_data_format() == 'channels_first':
            input_shape = (1, img_rows, img_cols)
        else:
            input_shape = (img_rows, img_cols, 1)

        ## モデル作成
        self.model = Sequential()
        # 畳み込み層1
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        # 最大プーリング層
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # 畳み込み層2
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        # 最大プーリング層
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
    
        self.model.add(Flatten())
        # 全結合層1
        self.model.add(Dense(NUM_H1))
        self.model.add(Activation('relu'))
        # 全結合層2
        self.model.add(Dense(NUM_CATEGORY))
        self.model.add(Activation('softmax'))

        ## モデルコンパイル(損失関数，最適化アルゴリズム，等の設定)
        #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model.summary()


    def fit(self, data_x, data_t, max_epochs):
        """
        print('self.model.predict(data_x)')
        print(self.model.predict(data_x[:200],batch_size=200)[0])
        print('self.model.predict(data_x).shape')
        print(self.model.predict(data_x[0:200],batch_size=200).shape)
        
        print('data_x[0]')
        print(data_x)
        print(data_x[:1].shape)
        print(data_x[:1])
        print(data_x[0].shape)
        print(data_x[0])
        print('self.model.predict(data_x[0]')
        print(self.model.predict(data_x[:1]))
        """
        #self.result = self.model.fit(data_x, data_t, batch_size=200, verbose=1, epochs=max_epochs, validation_split=0.1)
        #self.history = self.result.history
        #return self.result

    def load_weights(self, filename):
        # もし学習を続きから行いたい場合は，途中の重みファイルを指定する
        self.model.load_weights(filename)

    def save_weights(self, filename):
        self.model.save_weights(filename, True)

    def load_history(self, filename):
        with open(filename, mode='rb') as f:
            self.history = pickle.load(f)

    def save_history(self, filename):
        with open(filename, mode='wb') as f:
            pickle.dump(self.history, f)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=1)

    def plot_history(self):
        ## 学習時の誤差
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
        axL.plot(self.history['loss'],label="loss for training")
        axL.plot(self.history['val_loss'],label="loss for validation")
        axL.set_title('model loss')
        axL.set_xlabel('epoch')
        axL.set_ylabel('loss')
        axL.legend(loc='upper right')

        ## 学習時の精度
        axR.plot(self.history['acc'],label="accuracy for training")
        axR.plot(self.history['val_acc'],label="accuracy for validation")
        axR.set_title('model accuracy')
        axR.set_xlabel('epoch')
        axR.set_ylabel('accuracy')
        axR.legend(loc='lower right')

        plt.show()

def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    print(X_train.shape)
    """
    print(X_train[0].shape)
    """
    X_train, y_train = common2.convertCNNInput(X_train, y_train)
    X_test, y_test = common2.convertCNNInput(X_test, y_test)

    print(X_train.shape)
    print(type(X_train))
    ## 学習回数
    max_epochs = 15

    model = CNN()

    l = [8,9]
    a = np.array(l)
    print(a.shape)
    l = [[8,9],[8,9]]
    a = np.array(l)
    print(a.shape)
    l = [[[8,9],[8,9]]]
    a = np.array(l)
    print(a.shape)
    l = [[[[8,9],[8,9]]]]
    a = np.array(l)
    print(a.shape)


    print
    
    """
    model.fit(X_train, y_train, max_epochs)

    l = [[['a']],'b','c']
    print(l[0])
    print(l[:1])

    model.save_weights('cnn.hd5')

    model.save_history('cnn_history_15.pickle')


    ## テストデータに対する予測(精度)
    score = model.evaluate(X_test, y_test)
    print('accuracy for test data: ', score[1])

    #model.load_history('cnn_history_10to15.pickle')
    model.plot_history()
    """

if __name__ == '__main__':
    main()
