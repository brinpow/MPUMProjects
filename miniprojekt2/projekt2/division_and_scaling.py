import numpy as np


def scaling(data, size, type_):
    X = np.array(data[:size])
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


def division(scal=0):
    data = []
    with open("../resources/data/rp.data", "r") as data_file:
        for line in data_file:
            data_row = [int(x) for x in line.split()]
            data.append(data_row)

    data_zero = [x for x in data if x[9] == 2]
    data_one = [x for x in data if x[9] == 4]
    np.random.shuffle(data_one)
    np.random.shuffle(data_zero)

    size_zero = int(2*len(data_zero)/3)
    size_one = int(2*len(data_one)/3)
    data = data_zero[:size_zero] + data_one[:size_one] + data_zero[size_zero:] + data_one[size_one:]

    size = size_one + size_zero

    if scaling:
        scaling(data, size, scal)

    with open("../resources/data/training.data", "w") as train_data_file:
        for i in range(size):
            train_data_file.writelines([str(x)+" " for x in data[i]])
            train_data_file.write("\n")

    with open("../resources/data/testing.data", "w") as test_data_file:
        for i in range(size, len(data)):
            test_data_file.writelines([str(x)+" " for x in data[i]])
            test_data_file.write("\n")