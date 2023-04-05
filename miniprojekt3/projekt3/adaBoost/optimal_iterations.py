import matplotlib.pyplot as plt
import projekt3.prepare as pd
import adaBoost as ab
import projekt3.division_and_scaling as ds


def trial():
    ds.division(valid=True)
    X_train, y_train = pd.prepare_data("../resources/data/validation.data")
    X_test, y_test = pd.prepare_data("../resources/data/testing.data")

    cur_size = int(X_train.shape[0])
    X_present = X_train[:cur_size]
    y_present = y_train[:cur_size]

    values = ab.find_iter(X_present, y_present, X_test, y_test, 1000)

    return values


if __name__ == '__main__':
    val_sum = trial()
    for _ in range(9):
        val_cur = trial()
        val_sum += val_cur

    val_mean = val_sum/10

    with open("../resources/adaBoost/minims/iters.txt", "a") as result_file:
        result_file.write(f"Mean cost for iters is equal {val_mean}\n")

    plt.plot(val_mean, 'ro-')
    plt.xlabel("Iteracje")
    plt.ylabel("Błąd")
    plt.title("AdaBoost")
    plt.show()
