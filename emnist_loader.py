# To do for emnist_loader:
#  - Provide meta-data about number of characters in list
#  - Provide meta-data about number of pixels in each sample image

from emnist_byclass_mapping import byclass_map
from matplotlib import pyplot as plt
from mlxtend.data import loadlocal_mnist
import numpy as np
import os
from scipy.ndimage import zoom
from pprint import pprint

def gen_image(arr, width):
    two_d = (np.reshape(arr, (width, width)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt

# Load test and training data. "x" represents inputs, "y" represents outputs.
def load(emnist_path, width, data_char_set = ['x', 'o']):
    training_x, training_y = load_data_pair(
        os.path.join(emnist_path,'emnist-byclass-train-images-idx3-ubyte'),
        os.path.join(emnist_path,'emnist-byclass-train-labels-idx1-ubyte'),
        width,
        data_char_set
    )

    test_x, test_y = load_data_pair(
        os.path.join(emnist_path,'emnist-byclass-test-images-idx3-ubyte'),
        os.path.join(emnist_path,'emnist-byclass-test-labels-idx1-ubyte'),
        width,
        data_char_set
    )

    return training_x, training_y, test_x, test_y


def load_data_pair(images_path, labels_path, width, data_char_set):
    X, Y = loadlocal_mnist(images_path, labels_path)

    data_char_set_size = len(data_char_set)

    data_char_indices = [byclass_map[ord(x)] for x in data_char_set]

    indices = np.argwhere(Y == data_char_indices[0])    
    for i in range(1, data_char_set_size):
        indices = np.append(indices, np.argwhere(Y == data_char_indices[i]))

    np.random.shuffle(indices)

    X_list = [(zoom(arr.reshape(28, 28), (width/28.0))).flatten() for arr in X[indices]]
    X_chars = np.array(X_list)

    Y_chars = Y[indices]

    one_hot_lookup = {}

    for i in range(0, data_char_set_size):
        array = np.zeros(data_char_set_size, int).flatten()
        array[i] = 1
        one_hot_lookup[data_char_indices[i]] = array

    Y_one_hot = np.array([one_hot_lookup[n] for n in Y_chars])

    return X_chars, Y_one_hot


def main():
    width = 8

    emnist_path = os.path.join(os.getcwd(), 'emnist_data')

    X_ucf, Y_ucf, z, zz = load(emnist_path, width)

    print(Y_ucf[0:9])
    print('Dimensions: %s x %s' % (X_ucf.shape[0], X_ucf.shape[1]))

    print(Y_ucf[0])
    print('\n1st row', X_ucf[0])

    print(X_ucf.shape)

    fig = gen_image(X_ucf[0], width)
    fig.title(f'{Y_ucf[0]}')
    fig.show()

if __name__ == '__main__':
    main()
