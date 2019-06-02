import numpy as np
import numpy.linalg as la
from itertools import combinations
from operator import itemgetter


def linear_regression(x, y):
    padded_x = padd_x(x)
    w = np.dot(la.pinv(padded_x), y)
    return padded_x, w


def padd_x(x):
    return np.hstack((x, np.ones((x.shape[0], 1))))


def parity_function(x):
    power_set = []
    indexes = np.arange(len(x))

    for i in range(len(x)):
        power_set.append(combinations(indexes, i+1))

    output = [1]

    for power in power_set:
        for index in power:
            index = list(index)
            items = itemgetter(*index)(x)
            if len(index) > 1:
                output.append(np.prod(list(items)))
            else:
                output.append(items)

    return output


def extract_features(x):
    features = []
    for row in x:
        features.append(parity_function(row))
    features = np.array(features)
    return features


if __name__ == "__main__":
    x = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1]
    ])

    x_t = x.T

    y = np.array([-1, 1, 1, 1, -1, 1, 1, -1])

    padded_x, w = linear_regression(x, y)
    y_hat = np.dot(padded_x, w)
    print("Predictions 1: " , y_hat)

    features = extract_features(x)

    padded_features, w = linear_regression(features, y)
    y_hat = np.dot(padded_features, w)
    print("Predictions 2: ", y_hat)

