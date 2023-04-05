import matplotlib.pyplot as plt
import numpy as np
import projekt3.prepare as pd
import adaBoost as aB
import projekt3.division_and_scaling as ds

sizes = [0.005, 0.01, 0.02, 0.03, 0.075, 0.125, 0.25, 0.5, 0.625, 0.75, 0.875, 1]


def draw_graph(x, y, title):
    plt.plot(x, y, 'ro-')
    plt.xlabel("Frakcja danych")
    plt.ylabel(title)
    plt.title("AdaBoost")
    plt.show()


def trial():
    ds.division()
    X_train, y_train = pd.prepare_data("../resources/data/training.data")
    X_test, y_test = pd.prepare_data("../resources/data/testing.data")

    costs = []
    iters = []
    precisions = []
    tendernesses = []
    F1s = []

    with open("../resources/adaBoost/costs/cost.txt", "a") as result_file:
        for size in sizes:
            cur_size = int(X_train.shape[0]*size)
            X_present = X_train[:cur_size]
            y_present = y_train[:cur_size]

            cost_min, result, min_iter = aB.adaBoost(X_present, y_present, X_test, y_test, 1000)

            precision = result[3]/(result[3] + result[2])
            tenderness = result[3]/(result[3] + result[1])
            F1 = 2*precision*tenderness/(precision + tenderness)

            result_file.write(f"Cost for size: {size} is equal {cost_min}\n")
            result_file.write(f"Iterations are equal {min_iter}\n")
            result_file.write(f"Precision is equal {precision}\n")
            result_file.write(f"Tenderness is equal {tenderness}\n")
            result_file.write(f"F1 is equal {F1}\n")

            costs.append(cost_min)
            precisions.append(precision)
            tendernesses.append(tenderness)
            F1s.append(F1)
            iters.append(min_iter)
        result_file.write("\n")

    return np.array(costs), np.array(iters), np.array(precisions), np.array(tendernesses), np.array(F1s)


if __name__ == '__main__':
    costs_sum, iters_sum, precision_sum, tenderness_sum, F1_sum = trial()
    for _ in range(9):
        cost_cur, iters_cur, precision_cur, tenderness_cur, F1_cur = trial()
        costs_sum += cost_cur
        iters_sum += iters_cur
        precision_sum += precision_cur
        tenderness_sum += tenderness_cur
        F1_sum += F1_cur

    costs_mean = costs_sum/10
    iters_mean = iters_sum/10
    precision_mean = precision_sum/10
    tenderness_mean = tenderness_sum/10
    F1_mean = F1_sum/10

    with open("../resources/adaBoost/minims/minims.txt", "a") as result_file:
        result_file.write(f"Mean costs are equal {costs_mean}\n")
        result_file.write(f"Mean iterations are equal {iters_mean}\n")
        result_file.write(f"Mean precision is equal {precision_mean}\n")
        result_file.write(f"Mean tenderness is equal {tenderness_mean}\n")
        result_file.write(f"Mean F1 is equal {F1_mean}\n\n")

    draw_graph(sizes, costs_mean, "Błąd")
    draw_graph(sizes, iters_mean, "Iteracje")
    draw_graph(sizes, precision_mean, "Precyzja")
    draw_graph(sizes, tenderness_mean, "Czułość")
    draw_graph(sizes, F1_mean, "Miara F1")
