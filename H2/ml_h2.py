import numpy as np
import matplotlib.pyplot as plt

def train(x_train, y_train):
    return np.dot(np.linalg.pinv(x_train), y_train)

def generate_dataset(samples):
    x = np.random.rand(samples) * 2 - 1
    y = x ** 2
    x = np.array([[1, value] for value in x])
    return x, y

def inference(input, w):
    return np.dot(input, w)

def compute_error(results, y_train):
    return np.mean(np.power(results - y_train, 2))

def get_average_function_results(x, list_weights):
    results = inference(x, list_weights.transpose())
    return np.mean(results, axis=1)

def get_variance(x, list_weights, average_results, y_test):
    # results is NxM
    # for each row, a sample of the training set
    # for each column, a results of that sample with one the M classifiers
    results = np.dot(x, list_weights.transpose())
    # We need to compute the difference of the result of each classifier with the average classifier
    # if we duplicate the average classifier, we can just compute the difference between the results matrix
    # and the repeated average results
    average_results_repeated = np.array([average_results,]*results.shape[1]).transpose()

    return compute_error(average_results_repeated, results)

if __name__ == "__main__":

    num_classifiers = 1000

    list_weights = np.empty(shape=(num_classifiers,2))
    list_error = list()

    for iteration in range(num_classifiers):

        x_train, y_train = generate_dataset(samples=2)
        x_test, y_test = generate_dataset(samples=1000)

        w = train(x_train, y_train)

        results = inference(x_test, w)
        error = compute_error(results, y_test)

        list_weights[iteration][:] = w
        list_error.append(error)

    # compute Eout, bias and variance
    eout = np.mean(list_error)

    x_test, y_test = generate_dataset(samples=1000)

    average_results = get_average_function_results(x_test, list_weights)
    bias = compute_error(average_results, y_test)
    variance = get_variance(x_test, list_weights, average_results, y_test)
    print("Eout: " + str(eout))
    print("Bias: " + str(bias))
    print("Variance: " + str(variance))
    print("Bias+Variance: " + str(bias + variance))

    average_function = np.mean(list_weights, axis=0)

    x = np.arange(-1, 1.1, 0.1)
    f = x ** 2
    g = average_function[0] + average_function[1] * x

    plt.plot(x, f, x, g)
    plt.show()
