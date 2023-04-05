import numpy as np

DISTANCES = []


def euclidean_distance(x, y):
    sum = 0
    for i in range(len(x)):
        sum += (x[i]-y[i])*(x[i]-y[i])
    return sum


def hamming_distance(x, y):
    sum = 0
    for i in range(len(x)):
        if x[i]!=y[i]:
            sum += 1
    return sum


def find_distances(X_present, X_test, metric):
    global DISTANCES
    DISTANCES = []
    for i in range(X_test.shape[0]):
        DISTANCES.append([])
        for ii in range(X_present.shape[0]):
            if metric == 'Euclidean':
                dist = euclidean_distance(X_present[ii], X_test[i])
            else:
                dist = hamming_distance(X_present[ii], X_test[i])
            DISTANCES[i].append((dist, ii))

    for i in range(X_test.shape[0]):
        DISTANCES[i] = sorted(DISTANCES[i])


def find_neighbours(y_present, cur_x, k):
    distances = DISTANCES[cur_x][:k]

    sum_one = 0

    for dist, i in distances:
        if y_present[i] == 1:
            sum_one += 1

    if sum_one >= k - sum_one:
        return 1
    else:
        return -1


def find_k(X_present, y_present, X_test, y_test, metric='Euclidean'):
    values = []
    find_distances(X_present, X_test, metric)
    with open("../resources/k-nearest_neighbours/costs/k.txt", "a") as result_file:
        for i in range(1, int(np.sqrt(X_present.shape[0])) + 1):
            cur_cost, cur_result = cost(y_present, X_test, y_test, i)
            result_file.write(f"Cost for k: {i} is equal {cur_cost}\n")
            values.append(cur_cost)
        result_file.write("\n")
    return np.array(values)


def knn(X_present, y_present, X_test, y_test, metric='Euclidean'):
    best_cost = 1
    best_k = 0
    best_result = None
    find_distances(X_present, X_test, metric)
    for i in range(1, int(np.sqrt(X_present.shape[0]))+1):
        cur_cost, cur_result = cost(y_present, X_test, y_test, i)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_k = i
            best_result = cur_result
    return best_cost, best_result, best_k


def cost(y_present, X_test, y_test, k):
    result = [0, 0, 0, 0]
    for index in range(X_test.shape[0]):
        rv = find_neighbours(y_present, index, k)
        if rv == -1 and y_test[index] == -1:
            result[0] += 1  # TN
        elif rv == -1:
            result[1] += 1  # FN
        elif rv == 1 and y_test[index] == -1:
            result[2] += 1  # FP
        else:
            result[3] += 1  # TP
    return (result[1]+result[2])/X_test.shape[0], result