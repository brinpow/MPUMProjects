import numpy as np


def identity(X):
    return X


def quadratic(X):
    for i in range(X.shape[0]):
        for ii in range(1, X.shape[1]):
            X[i][ii] = X[i][ii]*X[i][ii]
    return X


def gaussian_function(X, s):
    means = np.mean(X, axis=0)
    for i in range(X.shape[0]):
        for ii in range(1, X.shape[1]):
            x_arg = -(X[i][ii]-means[ii])*(X[i][ii]-means[ii])/(2*s*s)
            X[i][ii] = np.exp(x_arg)
    return X


def sigmoid(X):
    means = np.mean(X, axis=0)
    for i in range(X.shape[0]):
        for ii in range(1, X.shape[1]):
            x_arg = (X[i][ii]-means[ii])*X[i][ii]
            X[i][ii] = 1/(1 + np.exp(-x_arg))
    return X


def remove_column(X, i):
    X = np.hstack((X[:, :i], X[:, i+1:]))
    return X


def sinus(X, alpha=12):
    for i in range(X.shape[0]):
        for ii in range(1, X.shape[1]):
            X[i][ii] = np.sin(alpha*X[i][ii])
    return X


def column_multiplication(X):
    new_X = []
    for i in range(X.shape[0]):
        row = []
        for ii in range(1, X.shape[1]):
            for j in range(ii, X.shape[1]):
                row.append(X[i][ii]*X[i][j])
        new_X.append(row)
    new_X = np.array(new_X)
    new_X = np.hstack((np.ones((new_X.shape[0], 1)), new_X))
    return new_X


def poly2nd(X):
    new_X = []
    for i in range(X.shape[0]):
        new_X.append([])
        for ii in range(0, X.shape[1]):
            for j in range(0, X.shape[1]):
                new_X[i].append(X[i][ii] * X[i][j])
    new_X = np.array(new_X)
    return new_X


def cube(X):
    for i in range(X.shape[0]):
        for ii in range(1, X.shape[1]):
            X[i][ii] = X[i][ii]*X[i][ii]*X[i][ii]
    return X


FUNCTIONS = [identity, quadratic, gaussian_function, sigmoid, remove_column, sinus, column_multiplication, poly2nd, cube]


def prepare_data(path, function_type, *args):
    with open(path, "r") as valid_data:
        data = []
        for line in valid_data:
            data_row = [np.double(x) for x in line[:-2].split(" ")]
            data.append(data_row)

    data = np.array(data)
    np.random.shuffle(data)
    data = np.hstack((np.ones((data.shape[0], 1)), data))

    X = data[:, :-1]
    y = data[:, -1].reshape((-1, 1))

    X = FUNCTIONS[function_type](X, *args)

    return X, y