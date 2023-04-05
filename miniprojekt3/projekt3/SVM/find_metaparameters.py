import matplotlib.pyplot as plt
import numpy as np
import projekt3.prepare as pd
import SMO
import projekt3.division_and_scaling as ds


def draw_graph(x, y, title):
    plt.plot(x, y, 'ro-')
    plt.xlabel("Frakcja danych")
    plt.ylabel(title)
    plt.title("AdaBoost")
    plt.show()


def trial():
    ds.division(valid=True)
    X_train, y_train = pd.prepare_data("../resources/data/validation.data")
    X_test, y_test = pd.prepare_data("../resources/data/testing.data")

    costs = []
    precisions = []
    tendernesses = []
    F1s = []
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    with open("../resources/SVM/costs/cost.txt", "a") as result_file:
        cur_size = int(X_train.shape[0])
        X_present = X_train[:cur_size]
        y_present = y_train[:cur_size]

        for C in Cs:
            alphas, param_b = SMO.SMO(X_present, y_present, "None", C, 500)
            cost_min, result = SMO.cost(X_test, y_test, X_present, y_present, alphas, param_b, "None")

            precision = result[3]/(result[3] + result[2])
            tenderness = result[3]/(result[3] + result[1])
            F1 = 2*precision*tenderness/(precision + tenderness)

            result_file.write(f"Cost for C: {C} is equal {cost_min}\n")
            result_file.write(f"Alphas are equal {alphas} and b {param_b}")
            result_file.write(f"Precision is equal {precision}\n")
            result_file.write(f"Tenderness is equal {tenderness}\n")
            result_file.write(f"F1 is equal {F1}\n")

            costs.append(cost_min)
            precisions.append(precision)
            tendernesses.append(tenderness)
            F1s.append(F1)
        result_file.write("\n")

    return np.array(costs), np.array(precisions), np.array(tendernesses), np.array(F1s)


if __name__ == '__main__':
    costs_sum, precision_sum, tenderness_sum, F1_sum = trial()
    for _ in range(0):
        cost_cur, precision_cur, tenderness_cur, F1_cur = trial()
        costs_sum += cost_cur
        precision_sum += precision_cur
        tenderness_sum += tenderness_cur
        F1_sum += F1_cur

    costs_mean = costs_sum/1
    precision_mean = precision_sum/1
    tenderness_mean = tenderness_sum/1
    F1_mean = F1_sum/1

    with open("../resources/SVM/minims/minims.txt", "a") as result_file:
        result_file.write(f"Mean costs are equal {costs_mean}\n")
        result_file.write(f"Mean precision is equal {precision_mean}\n")
        result_file.write(f"Mean tenderness is equal {tenderness_mean}\n")
        result_file.write(f"Mean F1 is equal {F1_mean}\n\n")

    draw_graph(sizes, costs_mean, "Błąd")
    #draw_graph(sizes, precision_mean, "Precyzja")
    #draw_graph(sizes, tenderness_mean, "Czułość")
    #draw_graph(sizes, F1_mean, "Miara F1")
