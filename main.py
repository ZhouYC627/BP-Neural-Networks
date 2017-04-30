#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from time import time
import math
'''
import matplotlib.pyplot as plt
data = np.genfromtxt('train_data.csv', delimiter=',')
for i in range(10):
    x = data[i]
    plt.imshow(x.reshape((20,20),order='F'),cmap='gray')
    plt.show()
'''

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def makeMatrix(m, n, fill=0.0):
    matrix = []
    for i in range(m):
        matrix.append([fill]*n)
    matrix = 0.1*np.random.random((m,n))-0.05
    return matrix

class NeuralNetwork:
    def __init__(self, ni, nh, no):

        # set weights to random values
        self.ni = ni;
        self.nh = nh;
        self.no = no;
        self.w = makeMatrix(nh, no)
        self.v = makeMatrix(ni, nh)
        self.ao = [0.0]*no
        self.ai = [0.0]*nh

    def train(self, train, targets, learning_rate=0.1, epochs=100):
        last_error = 0.0
        new_error = 0.0
        for k in range(epochs):
            for i in range(len(train)):
                x = train[i]#ni
                tar = targets[i]
                #print(tar)
                b = np.array(sigmoid(np.dot(x,self.v)-self.ai))#nh
                y = np.array(sigmoid(np.dot(b,self.w)-self.ao))#no
                error = np.ravel(tar-y)

                for j in error:
                    new_error += j**2
                #print('error: ', new_error)
                g = dsigmoid(y)*(tar-y)#no
                temp = np.dot(self.w, np.mat(g).T)
                temp.shape = (1,self.nh)
                e = np.multiply(b*(np.ones((1,self.nh))-b),temp)#nh

                self.w = self.w + learning_rate*np.dot(np.mat(b).T, np.mat(g))
                self.v = self.v + learning_rate*np.dot(np.mat(x).T, np.mat(e))
                self.ao = self.ao - learning_rate*g
                self.ai = self.ai - learning_rate*e
                #input("pause\n")
            if abs(last_error-new_error)/len(train)<0.003:
                print('rounds: ', k)
                break
            print('error:' , new_error/len(train))
            last_error = new_error
            new_error = 0.0

    def predict(self, x):
        b = sigmoid(np.dot(x, self.v)-self.ai);
        y = sigmoid(np.dot(b, self.w)-self.ao);
        return np.argmax(y)

if __name__ == '__main__':
    data = np.genfromtxt('train_data.csv', delimiter=',')
    temp = np.genfromtxt('train_targets.csv', delimiter=',')
    #preprocess
    targets = []
    for x in temp:
        tar = [0.0]*10
        tar[int(round(x))] = 1.0#no
        targets.append(tar)
    #print(targets)
    nn = NeuralNetwork(400,100,10)
    nn.train(data,targets,0.1,500)
    test = np.genfromtxt('test_data.csv', delimiter=',')
    result = []
    for i in test:
        result.append(nn.predict(i))
    np.savetxt('test_predictions.csv', np.transpose(result), fmt='%i')
