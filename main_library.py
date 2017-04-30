#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

'''
import matplotlib.pyplot as plt
data = np.genfromtxt('train_data.csv', delimiter=',')
for i in range(10):
    x = data[i]
    plt.imshow(x.reshape((20,20),order='F'),cmap='gray')
    plt.show()
'''

ni = 400
nh1 = 512
nh2 = 512
no = 10

#init
model = Sequential()

#add hidden layer
model.add(Dense(input_dim=ni, output_dim = nh1, activation = 'relu'))
model.add(Dense(input_dim=nh1, output_dim = nh2, activation = 'relu'))
model.add(Dense(input_dim=nh2, output_dim = no, activation = 'softmax'))

#loss
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

data = np.genfromtxt('train_data.csv', delimiter=',')
temp = np.genfromtxt('train_targets.csv', delimiter=',')
#preprocess
targets = []
for x in temp:
    tar = [0.0]*10
    tar[int(round(x))] = 1.0#no
    targets.append(tar)

#train
model.fit(data, targets, epochs=15)

#test
test = np.genfromtxt('test_data.csv', delimiter=',')
predictions = np.argmax(model.predict(test), 1)
np.savetxt('test_predictions_library.csv', np.transpose(predictions), fmt='%i')

#print(predictions)
