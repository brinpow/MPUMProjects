import numpy as np


def find_parameters(data):
    params = np.ones((data.shape[1]-1, 10, 2))

    total_t = 0
    for row in data:
        if row[-1] == 2:
            total_t += 1
            for i in range(len(row)-1):
                params[i][row[i]-1][0] += 1
        else:
            for i in range(len(row)-1):
                params[i][row[i]-1][1] += 1

    for i in range(data.shape[1]-1):
        for ii in range(10):
            params[i][ii][0] /= (total_t + 10)
            params[i][ii][1] /= (data.shape[0] - total_t + 10)

    phi_y = (total_t + 1)/(data.shape[0] + 2)

    return params, phi_y


def predict(val, params, phi_y):
    result = 0
    for i in range(len(val)-1):
        result += np.log(params[i][val[i]-1][0]/params[i][val[i]-1][1])

    result += np.log(phi_y/(1-phi_y))

    if result > 0:
        return True
    return False


def cost(data, params, phi_y):
    result = [0] * 4

    for val in data:
        rv = predict(val, params, phi_y)

        if rv and val[-1] == 2:
            result[0] += 1  # TN
        elif rv:
            result[1] += 1  # FN
        elif not rv and val[-1] == 2:
            result[2] += 1  # FP
        else:
            result[3] += 1  # TP

    return (result[1] + result[2])/data.shape[0], result
