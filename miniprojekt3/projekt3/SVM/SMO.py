import numpy as np

K = []
E = []


def gaussian(x, y, sigma):
    return np.exp(-(np.linalg.norm(x-y))**2 / (2 * sigma ** 2))


def linear(x, y):
    return np.dot(x.transpose(), y)


def sigmoid(x, y, alpha, beta):
    return np.tanh(alpha*np.dot(x.transpose(),y)+beta)


kernel_type = {"Linear": linear, "Gaussian": gaussian, "Sigmoid": sigmoid}


def create_kernel_matrix(X_1, X_2, type, *args):
    global K
    K = np.zeros((X_1.shape[0], X_2.shape[0]))
    for i in range(X_1.shape[0]):
        for ii in range(X_2.shape[0]):
            K[i][ii] = kernel_type[type](X_1[i], X_2[ii], *args)


def create_error_matrix(X, y, alphas, param_b):
    global E
    E = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        E[i] = np.dot(alphas * y.reshape(-1), K[:, i]) + param_b - y[i]


def SMO(X, y, type="Gausian", C=1, max_iter=1E3, *args):

    create_kernel_matrix(X, X, type, *args)
    alphas = np.zeros(X.shape[0])
    param_b = 0

    create_error_matrix(X, y, alphas, param_b)
    count = 0

    while True:
        a, b = np.random.choice(X.shape[0], 2, replace=False)

        alpha1 = alphas[a]
        alpha2 = alphas[b]

        Ea = E[a]
        Eb = E[b]

        eta = K[a][a] + K[b][b] - 2 * K[a][b]
        xi = -alpha1*y[a] - alpha2*y[b]
        s = y[a]*y[b]

        if eta == 0:
            continue

        if y[a] != y[b]:
            L = max(xi*y[b], 0)
            H = min(C + xi*y[b], C)
        else:
            L = max(0, -C - xi*y[b])
            H = min(C, -xi*y[b])

        a1 = alpha1 + y[a] * (Eb - Ea) / eta

        if a1 < L:
            a1_clip = L
        elif a1 > H:
            a1_clip = H
        else:
            a1_clip = a1

        if eta < 0:
            print("Weird")

        a2 = alpha2 + s * (alpha1 - a1_clip)

        b1 = -Ea + param_b + y[a]*(-a1_clip + alpha1) * K[a][a] + y[b]*(-a2 + alpha2) * K[a][b]
        b2 = -Eb + param_b + y[a]*(-a1_clip + alpha1) * K[a][b] + y[b]*(-a2 + alpha2) * K[b][b]

        if 0 < a1_clip < C:
            new_b = b1
        elif 0 < a2 < C:
            new_b = b2
        else:
            new_b = param_b

        alphas[a] = a1_clip
        alphas[b] = a2

        for i in range(X.shape[0]):
            E[i] = E[i] + y[a]*(a1_clip - alpha1) * K[a][i] + y[b]*(a2 - alpha2) * K[b][i] - param_b + new_b

        param_b = new_b
        count += 1

        if count > max_iter:
            break
    return alphas, param_b


def decision_function(z, y_train, alphas, param_b):
    rv = np.dot(alphas*y_train.reshape(-1), K[:, z]) + param_b
    if rv < 0:
        return -1
    return 1


def cost(X_test, y_test, X_train, y_train, alphas, param_b, type="Gaussian", *args):
    create_kernel_matrix(X_train, X_test, type, *args)
    result = [0, 0, 0, 0]
    for i in range(X_test.shape[0]):
        rv = decision_function(i, y_train, alphas, param_b)
        if rv == -1 and y_test[i] == -1:
            result[0] += 1  # TN
        elif rv == -1:
            result[1] += 1  # FN
        elif rv == 1 and y_test[i] == -1:
            result[2] += 1  # FP
        else:
            result[3] += 1  # TP
    return (result[1] + result[2]) / X_test.shape[0], result

