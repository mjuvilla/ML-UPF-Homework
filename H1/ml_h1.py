import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from datetime import datetime

max_value = 6

def classify(w, sample):
    return (np.sign(np.dot(w, sample)))

def generate_dataset(num_data_points, dimension):
    # generate x0 of each data point (always 1)
    x0 = np.ones(shape=(num_data_points, 1))

    # generate x1..xN
    data_points = 2 * max_value * np.random.random(size=(num_data_points, dimension)) - max_value

    # concatenate them
    return np.concatenate((x0, data_points), axis=1)

def plot_data(f, data_points, labels, w):
    x = np.array([-max_value, max_value])

    # compute the g classifier boundary
    f_line = - (f[0] + x * f[1]) / f[2]
    plt.plot(x, f_line, label="f")

    # compute the f classifier boundary
    if w is not None:
        w_line = - (w[0] + x * w[1]) / w[2]
        plt.plot(x, w_line, label="g")

    plt.legend()

    # find the positive examples (label = 1) and negative examples (label = -1)
    positive_examples = [idx for idx, label in enumerate(labels) if label == 1.0]
    negative_examples = [idx for idx, label in enumerate(labels) if label == -1.0]

    # plot them
    plt.plot(data_points[positive_examples, 1], data_points[positive_examples, 2], "go")
    plt.plot(data_points[negative_examples, 1], data_points[negative_examples, 2], "rx")
    # change the plot max values (x and y)
    plt.axis([-max_value, max_value, -max_value, max_value])
    plt.show()

def generate_random_f(data_points, dimension):

    # generate a boundary plane and check that it's inside our zone of interest
    while True:
        f = np.random.random(dimension+1) - 0.5
        y_value = - (f[0] + 0 * f[1]) / f[2]

        # if the value at 0 is inside de range (-max_value, max_value), it's good enough
        if (abs(y_value) <= max_value):
            break

    # generate the labels for the given f
    labels = [classify(f, sample) for sample in data_points]

    if plot_data_flag & (dimension == 2):
        plot_data(f, data_points, labels, None)

    return f, labels

def train_perceptron(data_points, labels, dimension):

    start = datetime.now()

    # random initialization
    w = np.random.random(dimension + 1) - 0.5
    steps = 0
    while True:
        correction = False
        for idx, data in enumerate(data_points):
            # if there's a mistake, try to correct it
            if classify(w, data) != labels[idx]:
                steps += 1
                w += labels[idx] * data
                correction = True

        # if there are no more errors, break
        if correction == False:
            break

    time_diff = datetime.now() - start
    time_diff_ms = time_diff.total_seconds() * 1000

    print("Finished training in " + "{0:.5f}".format(time_diff_ms) + " milliseconds " + str(steps) + " training steps.")

    return w, time_diff_ms, steps

def run(num_data_points, dimension=2):

    data_points = generate_dataset(num_data_points, dimension)

    f, labels = generate_random_f(data_points, dimension)

    w, train_time, steps = train_perceptron(data_points, labels, dimension)

    if plot_data_flag & (dimension == 2):
        plot_data(f, data_points, labels, w)

    return train_time, steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play with a perceptron.')
    parser.add_argument("num_data_points", type=int,
                        help='num of data points to be generated')
    parser.add_argument("--D", '--dimension', dest='dimension', type=int,
                        help='space dimension')
    parser.add_argument("--I", '--iterations', dest='iterations', type=int,
                        help='iterations', default=1)

    args = parser.parse_args()

    if args.iterations > 1:
        plot_data_flag = False
    else:
        plot_data_flag = True

    time_list = np.zeros(shape=args.iterations)
    steps_list = np.zeros(shape=args.iterations)

    for iteration in range(args.iterations):

        if args.dimension:
            train_time, steps = run(args.num_data_points, args.dimension)
        else:
            train_time, steps  = run(args.num_data_points)

        time_list[iteration] = train_time
        steps_list[iteration] = steps

    print()
    print("Average training time: " + str(time_list.mean()) + " and variance: " + str(time_list.var()))
    print("Average steps: " + str(steps_list.mean()) + " and variance: " + str(steps_list.var()))