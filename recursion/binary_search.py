# binary_search.py

import random
import time

import matplotlib.pyplot as plt

import numpy as np


def binary_search_recursive(data, target, low, high):

    if low > high:
        return False
    else:
        mid = (low + high) // 2

        if target == data[mid]:
            return True
        elif target < data[mid]:
            return binary_search_recursive(data, target, low, mid - 1)
        else:
            return binary_search_recursive(data, target, mid + 1, high)


def binary_search(data, target):
    return binary_search_recursive(data, target, data[0], data[len(data) - 1])


def linear_search(data, target):

    for el in data:
        if el == target:
            return True
    return False


def initialize_graph_data(graph_data):
    """
    generate data_sizes --> length(n), step_size(n)
    for size in data_sizes:
        generate list --> length(size)
        generate targets from list
        for target in targets:
            binary --> for average_size:
                            search for target
                        compute average --> avg_bs
            linear --> for average_size:
                            search for target
                        compute average --> avg_ls
            times.append([avg_bs, avg_ls])
        transmute data
        graph data
    """
    for size in graph_data["data_sizes"]:
        list_length_upperbound = size
        list_length_lowerbound = 0

        searchable_data = list(range(list_length_lowerbound, list_length_upperbound))

        amounts_of_targets = 10
        targets = []
        step_size = int(len(searchable_data) / amounts_of_targets)
        for i in range(0, len(searchable_data), step_size):
            targets.append(searchable_data[i])

        average_data = []

        for target in targets:
            start = time.time()
            binary_search(searchable_data, target)
            end = time.time()
            total_time = end - start
            average_data.append(total_time)

        graph_data["binary_times"].append(np.average(average_data))

        average_data = []

        for target in targets:
            start = time.time()
            linear_search(searchable_data, target)
            end = time.time()
            total_time = end - start
            average_data.append(total_time)

        graph_data["linear_times"].append(np.average(average_data))


def logFit(x, y):
    # cache some frequently reused terms
    sumy = np.sum(y)
    sumlogx = np.sum(np.log(x))

    b = (x.size * np.sum(y * np.log(x)) - sumy * sumlogx) / (
        x.size * np.sum(np.log(x) ** 2) - sumlogx**2
    )
    a = (sumy - b * sumlogx) / x.size

    return a, b


def logFunc(x, a, b):
    return a + b * np.log(x)


def main():
    data_size_lowerbound = 100
    amount_of_data_points = 1000
    data_size_step_size = 50
    data_size_upperbound = amount_of_data_points * data_size_step_size

    graph_data = {
        "data_sizes": list(range(data_size_lowerbound, data_size_upperbound)),
        "binary_times": [],
        "linear_times": [],
    }

    initialize_graph_data(graph_data)

    X = np.array(graph_data["data_sizes"])
    Y = np.array(graph_data["binary_times"])

    plt.plot(X, Y, ls="none", marker=".")
    xfit = np.linspace(0, data_size_upperbound, num=200)
    plt.plot(xfit, logFunc(xfit, *logFit(X, Y)))

    Y = np.array(graph_data["linear_times"])

    k, d = np.polyfit(X, Y, 1)
    y_pred = k * X + d
    plt.plot(X, Y, ls="none", marker=".")
    plt.plot(X, y_pred)

    plt.title("average search time vs data set size")
    plt.xlabel("data set size")
    plt.ylabel("average search time")
    plt.show()


if __name__ == "__main__":
    # This block of code will only be executed if you run the script from the terminal
    main()
