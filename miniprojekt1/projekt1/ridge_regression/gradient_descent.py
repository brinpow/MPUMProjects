import numpy as np


def gradient_descent(X, y, eta, batch_size, _lambda, iterations):
    data = np.hstack((X, y))
    theta = np.zeros((X.shape[1], 1))
    pos = 0
    for _ in range(iterations):
        if X.shape[0]//batch_size <= pos:
            pos = 0
        X_sam, y_sam = create_sample(data, batch_size, pos)
        theta = theta - eta*gradient_ols(X_sam, y_sam, theta, _lambda)
        pos += 1
    return theta


def gradient_ols(X, y, theta, _lambda):
    dif = (np.dot(X, theta) - y)
    grad = np.dot(1/X.shape[0] * X.transpose(), dif)
    grad += np.dot(_lambda*np.identity(theta.shape[0]), theta)
    grad = 2*grad
    return grad


def risk(X, y, theta):
    dif = np.dot(X, theta) - y
    J = np.dot(dif.transpose(), dif)
    J /= X.shape[0]
    return J[0]


def create_sample(data, batch_size, pos):
    if pos == 0:
        np.random.shuffle(data)
    data_sample = data[pos*batch_size:(pos+1)*batch_size, :]
    X_sample = data_sample[:, :-1]
    y_sample = data_sample[:, -1].reshape(-1, 1)
    return X_sample, y_sample
