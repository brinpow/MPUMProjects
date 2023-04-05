import cord_descent_net as cd
import numpy as np
import prepare_data as pd


def find_lambda(function_type, *args):
    X_valid, y_valid = pd.prepare_data("../resources/data/validation.data", function_type, *args)
    X_test, y_test = pd.prepare_data("../resources/data/testing.data", function_type, *args)

    lambda_values = 10 ** np.linspace(2, -9, 15) * 0.5
    alpha_values = 10 ** np.linspace(2, -9, 15) * 0.5

    with open("../resources/elastic_net/lambda_net_logs/net_lambda_gaussian.txt", "a") as lbd_file:
        mmin = -1
        theta = 0
        mlbd = 0
        malpha = 0
        for lamda_val in lambda_values:
            for alpha_val in alpha_values:
                theta_min = cd.coord_descent(X_valid, y_valid, lamda_val, alpha_val, 500)
                cost_min = cd.risk(X_test, y_test, theta_min)
                lbd_file.write(f"Lambda: {lamda_val}\n")
                lbd_file.write(f"Alpha {alpha_val}\n")
                lbd_file.write(f"Cost: {cost_min}\n")
                lbd_file.write(f"Theta: {theta_min}\n")
                lbd_file.write("\n")
                if mmin==-1 or cost_min<mmin:
                    mmin = cost_min
                    theta = theta_min
                    mlbd = lamda_val
                    malpha = alpha_val

    with open("../resources/elastic_net/lambda_net_logs/net_lambda_min_gaussian.txt", "a") as lbd_min_file:
        lbd_min_file.write(f"Minimal lambda {mlbd}\n")
        lbd_min_file.write(f"Minimal alpha: {malpha}\n")
        lbd_min_file.write(f"Minimal cost: {mmin}\n")
        lbd_min_file.write(f"Minimal theta: {theta}\n\n\n")

    return mlbd, malpha
