from keras.datasets import mnist
import numpy as np

#load in the training and testing data from MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#create a matrix of weights for each neuron (10 neurons in this case)
W = np.random.randn(10, 784) * 0.01

