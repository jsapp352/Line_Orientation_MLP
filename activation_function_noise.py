# Adapted from code by Milo Spencer-Harper
# source: https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a

import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, array, random
from pprint import pprint

_input_range = (-10, 10)
_sigmoid_region = (-7, 7)
_non_sigmoid_tick_interval = 0.5
_sigmoid_tick_interval = 0.1
_points_per_sigmoid_tick = 10
_standard_deviation = 1.0

def plot_activation(inputs, outputs):
    plt.plot(inputs, outputs, '.')
    plt.title(f'Activation Function with Gaussian Noise (standard deviation = {_standard_deviation})')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.show()

def sigmoid(x):
    return 1 / (1 + exp(-x + random.normal(0, _standard_deviation, x.shape)))

def activation_test(input_range, sigmoid_region, non_sigmoid_tick_interval, sigmoid_tick_interval, points_per_sigmoid_tick):
    input_start, input_end = input_range
    sigmoid_start, sigmoid_end = sigmoid_region

    inputs = np.linspace(input_start, input_end, int((input_end - input_start) / non_sigmoid_tick_interval))

    for i in range (0, points_per_sigmoid_tick):
        inputs = np.append(inputs, np.linspace(sigmoid_start, sigmoid_end, int((sigmoid_end-sigmoid_start) / sigmoid_tick_interval)))

    outputs = sigmoid(inputs)

    return inputs, outputs

def main():
    inputs, outputs = activation_test(_input_range, _sigmoid_region, _non_sigmoid_tick_interval, _sigmoid_tick_interval, _points_per_sigmoid_tick)

    plot_activation(inputs, outputs)
    # pprint(inputs)
    # pprint(outputs)

if __name__ == '__main__':
    main()
