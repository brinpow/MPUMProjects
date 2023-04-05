import matplotlib.pyplot as plt
import projekt3.prepare as pd
import knn
import projekt3.division_and_scaling as ds


def trial():
    ds.division(valid=True)
    X_train, y_train = pd.prepare_data("../resources/data/validation.data")
    X_test, y_test = pd.prepare_data("../resources/data/testing.data")

    cur_size = int(X_train.shape[0])
    X_present = X_train[:cur_size]
    y_present = y_train[:cur_size]

    values = knn.find_k(X_present, y_present, X_test, y_test, metric="Hamming")

    return values


if __name__ == '__main__':
    val_sum = trial()
    for _ in range(9):
        val_cur = trial()
        val_sum += val_cur

    val_mean = val_sum/10

    with open("../resources/k-nearest_neighbours/minims/k.txt", "a") as result_file:
        result_file.write(f"Mean costs for ks are equal {val_mean}\n")

    plt.plot(val_mean, 'ro-')
    plt.xlabel("Wartość parametru K")
    plt.ylabel("Błąd")
    plt.title("K najbliższych sąsiadów")
    plt.show()
