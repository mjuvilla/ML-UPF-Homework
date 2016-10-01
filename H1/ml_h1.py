import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from datetime import datetime

def classify(w, sample):
    return (np.sign(np.dot(w, sample)))

def generate_dataset(num_data_points, dimension):
    # generate x0 of each data point (always 1)
    x0 = np.ones(shape=(num_data_points, 1))

    # generate x1..xN
    data_points = 2 * np.random.random(size=(num_data_points, dimension)) - 1

    # concatenate them
    return np.concatenate((x0, data_points), axis=1)

def plot_data(w, data_points, labels, f=None):
    x = np.array([-1, 1])

    w_line = - (w[0] + x * w[1]) / w[2]
    plt.plot(x, w_line)

    if f is not None:
        f_line = - (f[0] + x * f[1]) / f[2]
        plt.plot(x, f_line)

    positive_examples = [idx for idx, label in enumerate(labels) if label == 1.0]
    negative_examples = [idx for idx, label in enumerate(labels) if label == -1.0]

    plt.plot(data_points[positive_examples, 1], data_points[positive_examples, 2], "o")
    plt.plot(data_points[negative_examples, 1], data_points[negative_examples, 2], "x")
    plt.axis([-1, 1, -1, 1])
    plt.show()

def generate_random_f(data_points, dimension):

    while True:
        f = np.random.random(dimension+1) - 0.5
        cosa = - (f[0] + 0 * f[1]) / f[2]

        if (abs(cosa) <= 1):
            break

    labels = [classify(f, sample) for sample in data_points]

    if dimension == 2:
        plot_data(f, data_points, labels)

    return f, labels

def train_perceptron(data_points, labels, dimension):

    start = datetime.now()

    # random initialization
    w = np.random.random(dimension + 1) - 0.5
    steps = 0
    while True:
        correction = False
        for idx, data in enumerate(data_points):
            steps += 1
            if classify(w, data) != labels[idx]:
                w += labels[idx] * data
                correction = True

        # if there are no more errors, break
        if correction == False:
            break

    time_diff = datetime.now() - start

    print("Finished training in " + "{0:.5f}".format(time_diff.total_seconds() * 1000) + " milliseconds and " + str(steps) + " training steps")

    return w

def run(num_data_points, dimension=2):

    data_points = generate_dataset(num_data_points, dimension)

    f, labels = generate_random_f(data_points, dimension)

    w = train_perceptron(data_points, labels, dimension)

    plot_data(w, data_points, labels, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play with a perceptron.')
    parser.add_argument("num_data_points", type=int,
                        help='num of data points to be generated')
    parser.add_argument("--D", '--dimension', dest='dimension',
                        help='space dimension')
    args = parser.parse_args()

    if args.dimension:
        run(args.num_data_points, args.dimension)
    else:
        run(args.num_data_points)