import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU, merge, Flatten
from keras import optimizers
from keras import regularizers
import keras.backend as K
from sklearn.metrics import mean_squared_error
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from xgboost.sklearn import XGBRegressor
import scipy.stats as st
from sklearn.model_selection import GridSearchCV

from unittest import TestCase

class IndexerLSTM():

    model = None

    def __init__(self, length, numAttr, neurons, dropouts, activations):
        '''
        Method to assign variables when the class is called.
        :param length:  length of the input array - part of the definition of the first layer shape
        :param numAttr: number of attributes - second part of the definition of the first layer shape
        :param neurons: array of integers defining the number of neurons in each layer and the number of layers (by the length of the array) - Do not include the final layer
        :param dropouts: array of doubles (0 - 1) of length neurons - 1 defining the dropout at each level - do not include the final layer
        :param activations: array of strings of length neurons to define the activation of each layer - do not include the final layer
        '''
        self.length = length
        self.numAttr = numAttr
        self.neurons = neurons
        self.dropouts = dropouts
        self.activations = activations

        ###Check to see if the length of arrays are appropriate
        if len(neurons) != len(dropouts) + 1:
            print('The length of the dropouts array must be one less than the length of the neurons array')
            raise ValueError
        if len(neurons) != len(activations):
            print('The length of the activations array must be the same as the length of the neurons array')
            raise ValueError


    def buildModel(self):
        '''
        Method to build the model based on the inputs when the class is instantiated.
        :return:
        '''
        self.model = Sequential()
        self.model.add(LSTM(self.neurons[0],return_sequences=True, activation=self.activations[0], input_shape=(self.length,self.numAttr)))
        self.model.add(Dropout(self.dropouts[0]))
        for i in np.arange(start = 1, stop = len(self.neurons)):
            self.model.add(LSTM(self.neurons[i], return_sequences=True, activation = self.activations[i]))
            self.model.add(Dropout(self.dropouts[i-1]))

        ###Add the final layer
        self.model.add(LSTM(self.neurons[len(self.neurons)-1],return_sequences=True, activation = self.activations[len(self.activations)-1]))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='linear',))
        pass


    def compileModel(self, loss='mse', optimizer='adam', metrics=['accuracy']):
        '''
        Method to compile the model using the parameters given by the inputs to the method
        :param loss: the loss functions to use in the compilation
        :param optimizer: the optimizer to use in the compilation
        :param metrics: list of metrics to use in the compilation
        :param shuffle: boolean to shuffle values in fitting
        :return:
        '''
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        pass


    def fitModel(self, x, y, epochs, batchSize,validation_data, verbose=0):
        '''
        Method to fit the model to data provided
        :param xtrain: inputs to train on
        :param ytrain: outputs to train on
        :param epochs: number of epochs to train the model
        :param batchSize: size of batches on which to train the model
        :param verbose: boolean (0, 1) value to control the verbosity of the fitting
        :return:
        '''
        m = self.model.fit(x, y, nb_epoch=epochs, batch_size=batchSize, verbose=verbose,validation_data=validation_data )
        return m


    def predict(self, x):
        '''
        Method to provide predictions from the model
        :param x: values on which to predict
        :return: predicted values
        '''
        y = self.model.predict(x)
        return y

    def getParams(self):
        '''
        Method to return the parameters of the model
        :return: all parameters of the current model
        '''
        return self.model.get_config()
    
    def get_model(self):
        '''
        Method to return the model
        '''
        return self.model


class TestIndexerLSTM(TestCase):
    def test_init1(self):
        length = 2
        numAttr = 2
        neurons = [100, 23, 28, 5]
        dropouts = [0.1, 0.05, 0.03, 0.04]
        activations = ['relu']

        with self.assertRaises(ValueError):
            IndexerLSTM(length=length, numAttr=numAttr, neurons=neurons, dropouts=dropouts,
                            activations=activations)


    def test_init2(self):
        length = 2
        numAttr = 2
        neurons = [100, 23, 28, 5]
        dropouts = [0.1, 0.05, 0.03]
        activations = ['relu']

        with self.assertRaises(ValueError):
            IndexerLSTM(length=length, numAttr=numAttr, neurons=neurons, dropouts=dropouts,
                            activations=activations)


    # def test_buildModel(self):
    #     self.fail()
    #
    # def test_compileModel(self):
    #     self.fail()
    #
    # def test_fitModel(self):
    #     self.fail()
    #
    # def test_predict(self):
    #     self.fail()
    # def test_getParams(self):
    #     self.fail()

