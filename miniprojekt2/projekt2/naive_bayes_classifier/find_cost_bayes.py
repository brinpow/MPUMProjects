import matplotlib.pyplot as plt
import numpy as np
import projekt2.prepare as pd
import naive_bayes as nb
import projekt2.division_and_scaling as ds

sizes = [0.005, 0.01, 0.02, 0.03, 0.075, 0.125, 0.25, 0.5, 0.625, 0.75, 0.875, 1]


def trial():
    ds.division()
    X_train, y_train = pd.prepare_data("../resources/data/training.data")
    X_test, y_test = pd.prepare_data("../resources/data/testing.data")

    data_train = np.hstack((X_train, y_train))
    data_test = np.hstack((X_test, y_test))

    costs = []
    precisions = []
    tendernesses = []
    F1s = []

    with open("../resources/naive_bayes/naive_bayes_costs/cost_bayes.txt", "a") as result_file:
        for size in sizes:
            cur_size = int(data_train.shape[0]*size)
            data_present = data_train[:cur_size]

            params, phi_y = nb.find_parameters(data_present)
            cost_min, result = nb.cost(data_test, params, phi_y)

            precision = result[3]/(result[3] + result[2])
            tenderness = result[3]/(result[3] + result[1])
            F1 = 2*precision*tenderness/(precision + tenderness)

            result_file.write(f"Cost for size: {size} is equal {cost_min}\n")
            result_file.write(f"Precision is equal {precision}\n")
            result_file.write(f"Tenderness is equal {tenderness}\n")
            result_file.write(f"F1 is equal {F1}\n")
            result_file.write(f"Conditional probabilities are equal: {params}\n")
            result_file.write(f"P(y) is equal: {phi_y}\n")
            costs.append(cost_min)
            precisions.append(precision)
            tendernesses.append(tenderness)
            F1s.append(F1)
        result_file.write("\n")

    return np.array(costs), params, phi_y, np.array(precisions), np.array(tendernesses), np.array(F1s)


if __name__ == '__main__':
    costs_sum, params_sum, phi_y_sum, precision_sum, tenderness_sum, F1_sum = trial()
    for _ in range(999):
        cost_cur, params_cur, phi_y_cur, precision_cur, tenderness_cur, F1_cur = trial()
        costs_sum += cost_cur
        params_sum += params_cur
        phi_y_sum += phi_y_cur
        precision_sum += precision_cur
        tenderness_sum += tenderness_cur
        F1_sum += F1_cur

    costs_mean = costs_sum/1000
    params_mean = params_sum/1000
    phi_y_mean = phi_y_sum/1000
    precision_mean = precision_sum/1000
    tenderness_mean = tenderness_sum/1000
    F1_mean = F1_sum/1000

    with open("../resources/naive_bayes/naive_bayes_minims/minims_bayes.txt", "a") as result_file:
        result_file.write(f"Mean costs are equal {costs_mean}\n")
        result_file.write(f"Mean precision is equal {precision_mean}\n")
        result_file.write(f"Mean tenderness is equal {tenderness_mean}\n")
        result_file.write(f"Mean F1 is equal {F1_mean}\n")
        result_file.write(f"Conditional probabilities are equal: {params_mean}\n")
        result_file.write(f"P(y) is equal: {phi_y_mean}\n\n")

    plt.plot(sizes, costs_mean, 'ro-')
    plt.xlabel("Frakcja danych")
    plt.ylabel("Błąd")
    plt.title("Naiwny klasyfikator bayesowski")
    plt.show()
    plt.plot(sizes, precision_mean, 'ro-')
    plt.xlabel("Frakcja danych")
    plt.ylabel("Precyzja")
    plt.title("Naiwny klasyfikator bayesowski")
    plt.show()
    plt.plot(sizes, tenderness_mean, 'ro-')
    plt.xlabel("Frakcja danych")
    plt.ylabel("Czułość")
    plt.title("Naiwny klasyfikator bayesowski")
    plt.show()
    plt.plot(sizes, F1_mean, 'ro-')
    plt.xlabel("Frakcja danych")
    plt.ylabel("Miara F1")
    plt.title("Naiwny klasyfikator bayesowski")
    plt.show()
