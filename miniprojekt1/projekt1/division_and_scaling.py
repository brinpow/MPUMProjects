import numpy as np


def find_min_max_mean(index, data, choosen):
    dmin = data[0][index]
    dmax = data[0][index]
    dsum = 0

    for i in range(len(data)):
        if choosen[i]:
            if data[i][index] < dmin:
                dmin = data[i][index]
            if data[i][index] > dmax:
                dmax = data[i][index]
            dsum += data[i][index]
    return dmin, dmax, dsum/len(choosen)


def choose(chosen, mmax):
    while True:
        rand_index = np.random.randint(0, mmax)
        if not chosen[rand_index]:
            return rand_index


def scale(data, mean, max, min, std_dev, type=0):
    if type == 0:
        return np.round((data - mean)/(max - min), 8)
    elif type == 1:
        return np.round((data - min)/(max - min), 8)
    else:
        return np.round((data - mean) / std_dev, 6)


def find_standard_deviation(index, data, chosen):
    train_data = [x[index] for i, x in enumerate(data) if chosen[i]]
    train_data = np.array(train_data)
    return np.std(train_data)


def div_and_scale():
    data = []
    with open("../resources/data/dane.data", "r") as data_file:
        for line in data_file:
            data_row = [np.double(x) for x in line.split("	")]
            data.append(data_row)

    chosen = [0]*len(data)
    for i in range(int(6*len(data)/10)):
        chosen[choose(chosen, len(data))] = 1

    for i in range(6):
        std_dev = find_standard_deviation(i, data, chosen)
        mmin, mmax, mmean = find_min_max_mean(i, data, chosen)
        for ii in range(len(data)):
            data[ii][i] = scale(data[ii][i], mmean, mmax, mmin, std_dev, 0)

    with open("../resources/data/training.data", "w") as train_data_file:
        for i in range(len(data)):
            if chosen[i]:
                train_data_file.writelines([str(x)+" " for x in data[i]])
                train_data_file.write("\n")

    data = [x for index, x in enumerate(data) if not chosen[index]]

    chosen = [0] * len(data)
    for i in range(int(len(data)/2)):
        chosen[choose(chosen, len(data))] = 1

    with open("../resources/data/testing.data", "w") as test_data_file:
        with open("../resources/data/validation.data", "w") as valid_data_file:
            for i in range(len(data)):
                if chosen[i]:
                    test_data_file.writelines([str(x)+" " for x in data[i]])
                    test_data_file.write("\n")
                else:
                    valid_data_file.writelines([str(x) + " " for x in data[i]])
                    valid_data_file.write("\n")