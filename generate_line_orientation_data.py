# by Justin Sapp
# Summer 2019

import argparse
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

class squareGenerator():
    def __init__(self):
        self.squares = []

        self.squares.append('V 1 0 1 0\n')
        self.squares.append('V 0 1 0 1\n')
        self.squares.append('H 1 1 0 0\n')
        self.squares.append('H 0 0 1 1\n')
        self.squares.append('D 1 0 0 1\n')
        self.squares.append('D 1 0 0 1\n')

    def createList(self, length):
        return [random.choice(self.squares) for x in range(1, length)]

def main():
    dataset_size = _args.length

    square_generator = squareGenerator()

    dataset = square_generator.createList(dataset_size)

    with open(_args.filename, 'w') as f:
        f.writelines(dataset)

if __name__ == '__main__':
    main()
