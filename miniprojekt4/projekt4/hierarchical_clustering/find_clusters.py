import matplotlib.pyplot as plt
import numpy as np
import projekt4.prepare_data as pd
import hierarchical_clustering as hc


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
    method = input("Choose method ")
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
        classes[int(y[i]-1)].append(X[i])

    with open(f"../resources/hierarchical_clustering/dists/{path}", "a") as result_file:
        hier = hc.HierarchicalClustering()
        hier.fit(X, method,4)
        cluster_assign = hier.aclusters()
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
            plt.scatter(result_x, result_y)
            acc += accuracy(classes, result)

        plt.title(method)
        plt.show()
        print(f"Accuracy {acc/y.shape[0]}")
        result_file.write(f"{hc.DISTS}\n")
        plt.plot(hc.DISTS, 'ro-')
        plt.title(method)
        plt.show()
