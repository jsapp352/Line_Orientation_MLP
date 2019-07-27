# by Justin Sapp
# Summer 2019

import argparse
import numpy as np
from numpy import array
import random

parser = argparse.ArgumentParser(description='Create a line-orientation dataset.')
parser.add_argument('length', type=int,
                    help='number of data samples to create')
parser.add_argument('filename',
                    help='name of output file (ex: data.txt)')

_args = parser.parse_args()

# Square Data Numbering Format:
#
# Label is positioned at data_line[0]:
#
# Pixels are postioned at data_line[n] accordingly:
#
#             ________________
#            |       |       |
#            |   1   |   2   |
#            |_______|_______|
#            |       |       |
#            |   3   |   4   |
#            |_______|_______|
#
#
#         V (vertical line)
# Labels: H (horizontal line)
#         D (diagonal line)
#
# Each sample square will have two pixels of value 1 (on) and two pixels of value 0 (off).

def analogify(data):
    high_value_mean = 0.75
    low_value_mean = 0.25
    standard_deviation = 0.05

    label, values = data

    new_values = values * np.random.normal(high_value_mean, standard_deviation, values.shape)

    new_values += np.logical_xor(values, 1) * np.random.normal(low_value_mean, standard_deviation, values.shape)

    return (label, new_values)


class squareGenerator():
    def __init__(self):
        self.squares = []

        self.squares.append(('V', array([1, 0, 1, 0])))
        self.squares.append(('V', array([0, 1, 0, 1])))
        self.squares.append(('H', array([1, 1, 0, 0])))
        self.squares.append(('H', array([0, 0, 1, 1])))
        self.squares.append(('D', array([1, 0, 0, 1])))
        self.squares.append(('D', array([1, 0, 0, 1])))

    def createList(self, length):
        return [analogify(random.choice(self.squares)) for x in range(1, length)]

def main():
    dataset_size = _args.length

    square_generator = squareGenerator()

    dataset = square_generator.createList(dataset_size)

    dataset_output = []

    for label, values in dataset:
        # line = f"{label}"

        # for value in values:
        #     line = f"{line}, {values}"
        value_string = f"{values}"[1:-1]

        line = f"{label} {value_string}\n"

        dataset_output.append(line)

    with open(_args.filename, 'w') as f:
        f.writelines(dataset_output)

if __name__ == '__main__':
    main()
