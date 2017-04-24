#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from time import time
import math
'''
import matplotlib.pyplot as plt

#for x in data:
x = data[10]
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
    matrix = 0.4*np.random.random((m,n))-0.2
    #matrix = np.random.random((m,n))
    #print(matrix)
    return matrix

class NeuralNetwork:
    def __init__(self, ni, nh, no):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        # set weights to random values
        self.ni = ni;
        self.nh = nh;
        self.no = no;
        self.w = makeMatrix(nh, no)
        self.v = makeMatrix(ni, nh)
        self.ao = [0.5]*no
        self.ai = [0.5]*nh
        '''
        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.05)
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.05)
        '''
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

                for j in range(len(error)):
                    new_error += error[j]**2
                #print('error: ', new_error)
                g = dsigmoid(y)*(tar-y)#no
                e = b*(np.ones(1)-b)*np.dot(self.w, np.mat(g).T)#nh
                self.w = self.w + learning_rate*np.dot(np.mat(b).T, np.mat(g))
                self.v = self.v + learning_rate*np.dot(np.mat(x).T, np.mat(e))
                #print('ai', self.ai)
                self.ao = self.ao - learning_rate*g
                self.ai = self.ai - learning_rate*e
            if abs(last_error-new_error)<1.0:
                print('epochs: ', k)
                break
            print('error:' , new_error)
            last_error = new_error
            new_error = 0.0

    def predict(self, x):
        '''
            x = np.array(x)
            temp = np.ones(x.shape[0]+1)
            temp[0:-1] = x
            a = temp
            #for l in range(0, len(self.weights)):
            #    a = self.activation(np.dot(a, self.weights[l]))
        '''
        b = sigmoid(np.dot(x, self.v)-self.ai);
        y = sigmoid(np.dot(b, self.w)-self.ao);
        '''
        result = 0
        print(y.shape)
        for i in range(len(y)):
            if (y[0][i]>0.5):
                result = i
        return result
        '''
        return np.argmax(y)

if __name__ == '__main__':
    '''
    nn = NeuralNetwork(2,2,1)
    x = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0,1,1,1])
    nn.train(x,y,0.1,1000)
    for i in [[0,0], [0,1], [1,0], [1,1]]:
        print(i, nn.predict(i))
    '''
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
    np.savetxt('my_predictions.csv', np.transpose(result), fmt='%i')
