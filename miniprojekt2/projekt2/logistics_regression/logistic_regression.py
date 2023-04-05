import numpy as np


def gradient_descent(X, y, eta, iterations):
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))
    y = np.array([(x[0]-2)/2 for x in y]).reshape(-1, 1)
    data = np.hstack((X, y))
    theta = np.zeros((X.shape[1], 1))
    pos = 0
    for _ in range(iterations):
        if X.shape[0] <= pos:
            pos = 0
        X_sam, y_sam = create_sample(data, pos)
        theta = theta + eta*gradient(X_sam, y_sam, theta)
        pos += 1
    return theta


def gradient(X, y, theta):
    grad = np.zeros((X.shape[1], 1))
    dif = (y[0][0] - 1/(1 + np.exp(-np.dot(X, theta)[0][0])))
    for i in range(X.shape[1]):
        grad[i][0] += dif*X[0][i]
    return grad


def create_sample(data, pos):
    if pos == 0:
        np.random.shuffle(data)
    data_sample = data[pos:pos+1, :]
    X_sample = data_sample[:, :-1]
    y_sample = data_sample[:, -1].reshape(-1, 1)
    return X_sample, y_sample


def predict(X_val, theta, y_val):
    result = (1/(1 + np.exp(-np.dot(X_val, theta))))[0]

    if result < 0.5:
        return True
    return False


def cost(X, y, theta):
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))
    result = [0] * 4

    for index, val in enumerate(X):
        rv = predict(val, theta, y[index][0])

        if rv and y[index][0] == 2:
            result[0] += 1  # TN
        elif rv:
            result[1] += 1  # FN
        elif not rv and y[index][0] == 2:
            result[2] += 1  # FP
        else:
            result[3] += 1  # TP

    return (result[1] + result[2])/X.shape[0], result
