import numpy as np


def scaling(data, type_):
    X = np.array(data)
    means = np.mean(X, axis=0)
    std_devs = np.std(X, axis=0)
    maxs = np.amax(X, axis=0)
    mins = np.amin(X, axis=0)
    for i in range(X.shape[1] - 1):
        mean = means[i]
        std_dev = std_devs[i]
        min = mins[i]
        max = maxs[i]
        for ii in range(len(data)):
            if type_ == 1:
                data[ii][i] = (data[ii][i] - mean) / std_dev
            elif type_ == 2:
                data[ii][i] = (data[ii][i] - min) / (max - min)
            elif type_ == 3:
                data[ii][i] = (data[ii][i] - mean) / (max - min)


def prepare_data(path, scal=0):
    with open(path, "r") as data_file:
        data = []
        for line in data_file:
            data_row = [np.double(x) for x in line.split()]
            data.append(data_row)

    data = np.array(data)

    if scal:
        scaling(data, scal)

    #X = data
    X = data[:, :-1]
    y = data[:, -1]

    return X, y