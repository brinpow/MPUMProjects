import numpy as np
import projekt3.sample as smp


class Classifier:
    def __init__(self, feature, pos, direct):
        self.feature = feature
        self.pos = pos
        self.direct = direct

    def predict(self, sample):
        if self.direct == 'right':
            if sample.features[self.feature] >= self.pos:
                return 1
            return -1
        else:
            if sample.features[self.feature] < self.pos:
                return 1
            return -1

    def printer(self):
        print(f"Feature {self.feature} {self.pos} {self.direct}")


def decision_function(weights, classifiers):
    def result_function(sample):
        result = 0
        for i in range(len(classifiers)):
            result += weights[i]*classifiers[i].predict(sample)
        if result >= 0:
            return 1
        return -1
    return result_function


def prepare_samples(X):
    samples = []
    for i in range(X.shape[0]):
        features = []
        for ii in range(X.shape[1]):
            features.append(X[i][ii])
        sample = smp.Sample(features, X.shape[0])
        samples.append(sample)
    return samples


def cost(samples, y, function):
    result = [0, 0, 0, 0]
    for index, sample in enumerate(samples):
        rv = function(sample)
        if rv == -1 and y[index] == -1:
            result[0] += 1  # TN
        elif rv == -1:
            result[1] += 1  # FN
        elif rv == 1 and y[index] == -1:
            result[2] += 1  # FP
        else:
            result[3] += 1  # TP
    return (result[1]+result[2])/len(samples), result


def check_classifier(classifier, samples, y):
    correct = 0
    for index, sample in enumerate(samples):
        if classifier.predict(sample) == y[index]:
            correct += sample.weight
    return correct


def find_classifier(samples, y):
    classifiers = []

    def check(i, poses):
        nonlocal classifiers
        for ii in poses:
            cur_classifier = Classifier(i, ii - 1, 'right')
            cur_classifier2 = Classifier(i, ii - 1, 'left')
            rv = check_classifier(cur_classifier, samples, y)
            rv2 = check_classifier(cur_classifier2, samples, y)

            if rv > rv2:
                classifiers.append(cur_classifier)
            else:
                classifiers.append(cur_classifier2)

    for i in range(21):
        check(i, [-1, 1, 2])
    for i in range(21, 29):
        check(i, [-1, 0, 1, 2])
    check(29, [0, 1, 2])

    return classifiers


def find_best(classifiers, samples, y):
    best_classifier = None
    best_val = 0

    for classifier in classifiers:
        rv = check_classifier(classifier, samples, y)
        if rv>best_val:
            best_val = rv
            best_classifier = classifier
    return best_classifier, 1 - best_val


def adaBoost(X_train, y_train, X_test, y_test, iterations):
    samples_train = prepare_samples(X_train)
    samples_test = prepare_samples(X_test)
    classifiers = []
    alphas = []

    min_cost = 1
    min_result = 0
    min_iter = 0

    set_classifiers = find_classifier(samples_train, y_train)

    for i in range(iterations+1):
        classifier, epsilon = find_best(set_classifiers, samples_train, y_train)
        classifiers.append(classifier)
        alpha = 1/2*np.log((1-epsilon)/epsilon)
        alphas.append(alpha)
        zet = 2*np.sqrt(epsilon*(1-epsilon))

        sum = 0
        for ii in range(len(samples_train)):
            value = samples_train[ii].weight/zet*np.exp(-alpha*y_train[ii]*classifier.predict(samples_train[ii]))
            samples_train[ii].set_weight(value)
            sum += value

        for ii in range(len(samples_train)):
            samples_train[ii].set_weight(samples_train[ii].weight/sum)

        if i % 50 == 0:
            cur_cost, cur_result = cost(samples_test, y_test, decision_function(alphas, classifiers))
            if cur_cost<min_cost:
                min_cost = cur_cost
                min_iter = i
                min_result = cur_result
    return min_cost, min_result, min_iter


def find_iter(X_train, y_train, X_test, y_test, max_iter):
    values = []
    samples_train = prepare_samples(X_train)
    samples_test = prepare_samples(X_test)
    classifiers = []
    alphas = []

    set_classifiers = find_classifier(samples_train, y_train)

    with open("../resources/adaBoost/costs/iters.txt", "a") as result_file:
        for i in range(1,max_iter + 1):
            classifier, epsilon = find_best(set_classifiers, samples_train, y_train)
            classifiers.append(classifier)
            alpha = 1 / 2 * np.log((1 - epsilon) / epsilon)
            alphas.append(alpha)
            zet = 2*np.sqrt(epsilon*(1-epsilon))

            sum = 0
            for ii in range(len(samples_train)):
                value = samples_train[ii].weight/zet* np.exp(-alpha * y_train[ii] * classifier.predict(samples_train[ii]))
                samples_train[ii].set_weight(value)
                sum += value

            for ii in range(len(samples_train)):
                samples_train[ii].set_weight(samples_train[ii].weight / sum)

            if i % 50 == 0:
                cur_cost, cur_result = cost(samples_test, y_test, decision_function(alphas, classifiers))
                result_file.write(f"Cost for iters: {i} is equal {cur_cost}\n")
                values.append(cur_cost)
        result_file.write("\n")
    return np.array(values)