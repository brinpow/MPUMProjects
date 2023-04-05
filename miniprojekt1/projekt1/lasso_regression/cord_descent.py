import numpy as np


def check_range(c, lambda_, a):
    if c < -lambda_:
        return (c + lambda_)/a
    elif c > lambda_:
        return (c - lambda_)/a
    return 0


def coord_descent(X, y, lambda_, iterations):
    theta = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        for j in range(X.shape[1]):
            X_j = X[:, j].reshape(-1, 1)
            c = np.dot(X_j.transpose(), (y - np.dot(X, theta) + X_j*theta[j][0]))
            c = (2*c/X.shape[0])[0][0]
            a = (2/X.shape[0]*np.dot(X_j.transpose(), X_j))[0][0]
            theta[j][0] = check_range(c, lambda_, a)
    return theta


def risk(X, y, theta):
    dif = np.dot(X, theta) - y
    J = np.dot(dif.transpose(), dif)
    J /= X.shape[0]
    return J[0]
