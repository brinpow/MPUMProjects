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


def division(scall=0, valid=False):
    data = []
    with open("../resources/data/phishing.data", "r") as data_file:
        for line in data_file:
            data_row = [int(x) for x in line.split(',')]
            data.append(data_row)

    data_zero = [x for x in data if x[30] == -1]
    data_one = [x for x in data if x[30] == 1]
    np.random.shuffle(data_one)
    np.random.shuffle(data_zero)

    scal = 2/3
    if valid:
        scal = 3/5

    size_zero = int(len(data_zero)*scal)
    size_one = int(len(data_one)*scal)
    data = data_zero[:size_zero] + data_one[:size_one] + data_zero[size_zero:] + data_one[size_one:]

    size = size_one + size_zero

    if scall:
        scaling(data, size, scal)

    with open("../resources/data/training.data", "w") as train_data_file:
        for i in range(size):
            train_data_file.writelines([str(x)+" " for x in data[i]])
            train_data_file.write("\n")

    if not valid:
        with open("../resources/data/testing.data", "w") as test_data_file:
            for i in range(size, len(data)):
                test_data_file.writelines([str(x)+" " for x in data[i]])
                test_data_file.write("\n")
    else:
        size_test_zero = int((len(data_zero)-size_zero)/2)
        size_test_one = int((len(data_one)-size_one)/2)
        with open("../resources/data/testing.data", "w") as test_data_file:
            for i in range(size, size + size_test_zero):
                test_data_file.writelines([str(x)+" " for x in data[i]])
                test_data_file.write("\n")
            for i in range(size + len(data_zero)-size_zero, size + len(data_zero)-size_zero+size_test_one):
                test_data_file.writelines([str(x) + " " for x in data[i]])
                test_data_file.write("\n")
        with open("../resources/data/validation.data", "w") as test_data_file:
            for i in range(size + size_test_zero, size + len(data_zero)-size_zero):
                test_data_file.writelines([str(x) + " " for x in data[i]])
                test_data_file.write("\n")
            for i in range(size + len(data_zero)-size_zero+size_test_one, len(data)):
                test_data_file.writelines([str(x) + " " for x in data[i]])
                test_data_file.write("\n")