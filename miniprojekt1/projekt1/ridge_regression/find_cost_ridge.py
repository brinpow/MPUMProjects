import matplotlib.pyplot as plt
import gradient_descent as gd
import prepare_data as pd
import numpy as np
import find_lambda_ridge as fl
import division_and_scaling as ds

if __name__ == '__main__':
    function_type = int(input("Base function type: \n"))

    ds.div_and_scale()
    X_train, y_train = pd.prepare_data("../resources/data/training.data", function_type, 10)
    X_test, y_test = pd.prepare_data("../resources/data/testing.data", function_type, 10)

    costs = []
    sizes = [0.01, 0.02, 0.03, 0.075, 0.125, 0.25, 0.5, 0.625, 0.75, 0.875, 1]
    lambda_ = fl.find_lambda(function_type, 10)
    print(lambda_)
    with open("../resources/ridge/ridge_costs/ridge_cost_sinus.txt", "a") as result_file:
        result_file.write(f"Minimal lambda {lambda_}\n")
        for size in sizes:
            cur_size = int(X_train.shape[0]*size)
            batch_size = int(size*31) + 1
            X_present = X_train[:cur_size, :]
            y_present = y_train[:cur_size, :]
            theta_min = gd.gradient_descent(X_present, y_present, 0.01, batch_size, lambda_, 10000)
            cost_min = gd.risk(X_test, y_test, theta_min)[0]
            result_file.write(f"Cost for size: {size} is equal {cost_min}\n")
            result_file.write(f"Theta is equal: {theta_min}\n")
            costs.append(np.sqrt(cost_min))
        result_file.write("\n")

    plt.plot(sizes, costs, 'ro-')
    costs = [x*x for x in costs]
    print(costs)
    plt.show()
