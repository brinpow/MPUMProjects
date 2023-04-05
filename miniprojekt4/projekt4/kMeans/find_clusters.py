import matplotlib.pyplot as plt
import numpy as np
import projekt4.prepare_data as pd
import kMeans as km

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
    algo = input("Algorithm type ")
    if number<9:
        path = f"dane_2D_{number}.txt"
    elif number == 9:
        path = "dane_9D.txt"
    else:
        path = "rp.data"

    X, y = pd.prepare_data(f"../resources/data/{path}")

    classes = []
    for i in range(int(np.max(y))):
        classes.append([])
    for i in range(y.shape[0]):
        classes[int(y[i] - 1)].append(X[i])

    inertias = []
    accuracies = []

    with open(f"../resources/Kmeans/inertias/{path}", "a") as result_file:
        for ij in range(2, 3):
            Kmeans = km.Kmeans()
            iter, inertia = Kmeans.fit(X, ij, 300, algo)
            cluster_assign = Kmeans.clusters()

            acc = 0
            for i in np.unique(cluster_assign):
                result_x = []
                result_y = []
                result = []
                for ii in range(cluster_assign.shape[0]):
                    if cluster_assign[ii] == i:
                        result.append(X[ii])
                        result_x.append(X[ii][0])
                        result_y.append(X[ii][1])
                acc += accuracy(classes, result)
                plt.scatter(result_x, result_y)

            plt.title(algo)
            plt.show()
            inertias.append(inertia)
            accuracies.append(accuracies)
            result_file.write(f"Iterations {ij} accuracy {acc/y.shape[0]} iters {iter} inertia {inertia} ymax {np.max(y)}\n")
        result_file.write('\n')

    plt.title("Inertia")
    plt.plot(inertias, 'ro-')
    plt.show()

