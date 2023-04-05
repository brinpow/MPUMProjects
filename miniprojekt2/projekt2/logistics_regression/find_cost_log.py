import matplotlib.pyplot as plt
import numpy as np

import projekt2.prepare as pd
import logistic_regression as lr
import projekt2.division_and_scaling as ds

sizes = [0.005, 0.01, 0.02, 0.03, 0.075, 0.125, 0.25, 0.5, 0.625, 0.75, 0.875, 1]


def trial():
    ds.division(1)
    X_train, y_train = pd.prepare_data("../resources/data/training.data")
    X_test, y_test = pd.prepare_data("../resources/data/testing.data")

    costs = []
    precisions = []
    tendernesses = []
    F1s = []

    with open("../resources/logistic_regression/log_regression_costs/cost_bayes.txt", "a") as result_file:
        for size in sizes:
            cur_size = int(X_train.shape[0]*size)
            X_present = X_train[:cur_size]
            y_present = y_train[:cur_size]

            theta = lr.gradient_descent(X_present, y_present, 0.001, 10000)
            cost_min, result = lr.cost(X_test, y_test, theta)

            try:
                precision = result[3] / (result[3] + result[2])
            except ZeroDivisionError:
                precision = 0
            tenderness = result[3] / (result[3] + result[1])
            try:
                F1 = 2 * precision * tenderness / (precision + tenderness)
            except ZeroDivisionError:
                F1 = 0
            precisions.append(precision)
            tendernesses.append(tenderness)
            F1s.append(F1)

            result_file.write(f"Cost for size: {size} is equal {cost_min}\n")
            result_file.write(f"Precision is equal {precision}\n")
            result_file.write(f"Tenderness is equal {tenderness}\n")
            result_file.write(f"F1 is equal {F1}\n")
            result_file.write(f"Theta is equal equal: {theta}\n")
            costs.append(cost_min)
        result_file.write("\n")

    return np.array(costs), theta, np.array(precisions), np.array(tendernesses), np.array(F1s)


if __name__ == '__main__':
    costs_sum, theta_sum, precision_sum, tenderness_sum, F1_sum = trial()
    for _ in range(99):
        cost_cur, theta_cur, precision_cur, tenderness_cur, F1_cur = trial()
        costs_sum += cost_cur
        theta_sum += theta_cur
        precision_sum += precision_cur
        tenderness_sum += tenderness_cur
        F1_sum += F1_cur

    costs_mean = costs_sum/100
    theta_mean = theta_sum/100
    precision_mean = precision_sum / 100
    tenderness_mean = tenderness_sum / 100
    F1_mean = F1_sum / 100

    with open("../resources/logistic_regression/log_regression_minims/minims_log_regression.txt", "a") as result_file:
        result_file.write(f"Mean costs are equal {costs_mean}\n")
        result_file.write(f"Mean precision is equal {precision_mean}\n")
        result_file.write(f"Mean tenderness is equal {tenderness_mean}\n")
        result_file.write(f"Mean F1 is equal {F1_mean}\n")
        result_file.write(f"Theta is equal: {theta_mean}\n\n")

    plt.plot(sizes, costs_mean, 'ro-')
    plt.xlabel("Frakcja danych")
    plt.ylabel("Błąd")
    plt.title("Regresja logistyczna")
    plt.show()
    plt.plot(sizes, precision_mean, 'ro-')
    plt.xlabel("Frakcja danych")
    plt.ylabel("Precyzja")
    plt.title("Regresja logistyczna")
    plt.show()
    plt.plot(sizes, tenderness_mean, 'ro-')
    plt.xlabel("Frakcja danych")
    plt.ylabel("Czułość")
    plt.title("Regresja logistyczna")
    plt.show()
    plt.plot(sizes, F1_mean, 'ro-')
    plt.xlabel("Frakcja danych")
    plt.ylabel("Miara F1")
    plt.title("Regresja logistyczna")
    plt.show()
