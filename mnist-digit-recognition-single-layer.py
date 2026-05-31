from keras.datasets import mnist
import numpy as np

def softmax(z):
    e_z = np.exp(z)
    return e_z / np.sum(e_z)

def argmax(given_smax):
    max_ind = 0
    for i in range(0, len(given_smax)):
       if given_smax[i] > given_smax[max_ind]:
            max_ind = i

    return max_ind

#load in the training and testing data from MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784) / 255;
x_test = x_test.reshape(10000, 784) / 255;

def computeAccuracy(W, b):
    "returns a float that represents the percent accuracy of our model."

    correct_count = 0

    for i in range(0, len(x_test)):
        z = np.dot(W, x_test[i]) + b
        yhat = softmax(z)

        prediction = argmax(yhat)

        if prediction == y_test[i]:
            correct_count += 1

    return correct_count/len(x_test)


def computeModel():
    "returns W (weight matrix) and b (bias) for model"""

    #create a matrix of weights for each neuron (10 neurons in this case)
    W = np.random.randn(10, 784) * 0.01

    #bias too
    b = np.zeros(10);

    #learning rate
    alpha = 0.01

    #let's iterate through all of the training images
    for epoch in range(0, 50):

        loss_total = 0

        for image_i in range(0, len(x_train)):
            
            z = np.dot(W, x_train[image_i]) + b
            yhat = softmax(z)

            y = [0]*10
            y[y_train[image_i]] = 1

            loss_total += -np.sum(y * np.log(yhat))

            #compute derivatives dL/dW and dL/db.
            err = yhat - y
            dW = np.outer(err, x_train[image_i])
            db = err

            #train the parameters with those two derivaties we computed.
            W -= alpha * dW
            b -= alpha * db

        print(f'avg loss: {loss_total/len(x_train)}')

    return (W, b)

(W, b) = computeModel()

accuracy = computeAccuracy(W, b)

print(f'{accuracy*100}% accuracy.')
