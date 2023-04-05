import matplotlib.pyplot as plt
import numpy as np
import projekt4.prepare_data as pd
import spectral as sc

def accuracy(classes, result):
    max_acc = 0
    for clas in classes:
        acc = 0
        for res in result:
            for c in clas:
                equal = True
                for dim in range(len(res)):
                    if res[dim] != c[dim]:
                        equal = False
                if equal:
                    acc += 1
                    break
        if acc>max_acc:
            max_acc = acc
    return max_acc


if __name__ == '__main__':
    number = int(input("Choose file "))
    if number<9:
        path = f"dane_2D_{number}.txt"
    elif number == 9:
        path = "dane_9D.txt"
    else:
        path = "rp.data"

    X, y = pd.prepare_data(f"../resources/data/{path}", scal=3)

    classes = []
    for i in range(int(np.max(y))):
        classes.append([])
    for i in range(y.shape[0]):
        classes[int(y[i]-1)].append(X[i])

    inertias = []
    accuracies = []

    with open(f"../resources/spectral_clustering/inertias/{path}", "a") as result_file:
        for ij in range(1, 15):
            sp = sc.SpectralClustering(k=ij, sigma=1.0, epsilon=0.1)
            inertia = sp.fit(X, "Epsilon")
            cluster_assign = sp.aclusters()
            acc = 0
            for i in np.unique(cluster_assign):
                result_x = []
                result_y = []
                result = []
                for ii in range(cluster_assign.shape[0]):
                    if cluster_assign[ii] == i:
                        result_x.append(X[ii][0])
                        result_y.append(X[ii][1])
                        result.append(X[ii])
                acc += accuracy(classes, result)
                plt.scatter(result_x, result_y)

            inertias.append(inertia)
            accuracies.append(accuracies)
            result_file.write(f"Iterations {ij} accuracy {acc / y.shape[0]} inertia {inertia} ymax {np.max(y)}\n")
            result_file.write('\n')

    plt.title("Inertia")
    plt.plot(inertias, 'ro-')
    plt.show()
