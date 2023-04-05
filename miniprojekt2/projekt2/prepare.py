import numpy as np


def prepare_data(path):
    with open(path, "r") as data_file:
        data = []
        for line in data_file:
            data_row = [np.double(x) for x in line[:-2].split(" ")]
            data.append(data_row)

    data = np.array(data)
    np.random.shuffle(data)

    X = data[:, :-1]
    y = data[:, -1].reshape((-1, 1))

    return X, y