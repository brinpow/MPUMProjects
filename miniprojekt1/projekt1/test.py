import numpy as np
import matplotlib.pyplot as plt
import prepare_data as pd
import elasric_net_regression.cord_descent_net as cd

if __name__ == '__main__':

    for s in range(1, 21):
        X_valid, y_valid = pd.prepare_data("resources/data/validation.data", 2, s)
        X_test, y_test = pd.prepare_data("resources/data/testing.data", 2, s)

        lambda_values = 10 ** np.linspace(2, -9, 100) * 0.5
        allph = 10 ** np.linspace(2, -9, 100) * 0.5

        with open("resources/elastic_net/testing_metaparameters/test_gaussian.txt", "a") as lbd_file:
            mmin = -1
            theta = 0
            mlbd = 0
            malpha = 0
            ms = 0
            for lamda_val in lambda_values:
                for alpha_val in allph:
                    theta_min = cd.coord_descent(X_valid, y_valid, lamda_val, alpha_val, 1000)
                    cost_min = cd.risk(X_test, y_test, theta_min)
                    lbd_file.write(f"S: {s}")
                    lbd_file.write(f"Lambda: {lamda_val}\n")
                    lbd_file.write(f"Alpha: {alpha_val}\n")
                    lbd_file.write(f"Cost: {cost_min}\n")
                    lbd_file.write(f"Theta: {theta_min}\n")
                    lbd_file.write("\n")
                    if mmin == -1 or cost_min < mmin:
                        malpha = alpha_val
                        mmin = cost_min
                        theta = theta_min
                        mlbd = lamda_val
                        ms = s

        with open("resources/elastic_net/testing_metaparameters/mingaussian.txt", "a") as lbd_min_file:
            lbd_min_file.write(f"Minimal s {ms}\n")
            lbd_min_file.write(f"Minimal lambda {mlbd}\n")
            lbd_min_file.write(f"Minimal alpha {malpha}\n")
            lbd_min_file.write(f"Minimal cost: {mmin}\n")
            lbd_min_file.write(f"Minimal theta: {theta}\n\n\n")