import numpy as np;
import matplotlib.pyplot as plt;

import random;

def getAccuracy(m, b):
    "Compute the accuracy percentage given the m & b defining the regression line."

    correct_count = 0;

    for i in range(0, 100):
        #generate a test coord pair (there's probably a neater way to do this)
        x = random.uniform(0, 8);
        y = 0
        if x >= 4.5: y = 1

        #generate our prediction from the test coord x
        yhat = 1/(1 + np.exp((-m*x - b)))
        prediction = 0;
        if yhat >= 0.5:
            prediction = 1;

        if y == prediction:
            correct_count += 1

    return correct_count / 100



def computeRegression():
    "returns the slope and intercept of the regression function."

    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    ys = np.array([0  ,   0,   0,   0,   1,   1,   1,   1]);

    m, b = 0, 0
    alpha = 1

    for i in range(0, 100):

        yhat = 1/(1 + np.exp((-m*xs - b)))
        yhat = np.clip(yhat, 1e-7, 1 - 1e-7)

        cross_entropy = -np.mean(ys * np.log(yhat) + (1 - ys) * np.log(1-yhat));

        dm = np.mean((yhat - ys) * xs)
        db = np.mean(yhat - ys)

        m -= alpha*dm;
        b -= alpha*db;

    return (m, b)


if __name__ == '__main__':

    (m, b) = computeRegression();
    accuracy = getAccuracy(m, b)*100;

    print(f'model accuracy is {accuracy}%');
