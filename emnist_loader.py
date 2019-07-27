from matplotlib import pyplot as plt
from mlxtend.data import loadlocal_mnist
import numpy as np
import os
from scipy.ndimage import zoom

def gen_image(arr, width):
    two_d = (np.reshape(arr, (width, width)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt

def load(emnist_path, width):
    training_x, training_y = load_data_pair(
        os.path.join(emnist_path,'emnist-letters-train-images-idx3-ubyte'),
        os.path.join(emnist_path,'emnist-letters-train-labels-idx1-ubyte'),
        width
    )

    test_x, test_y = load_data_pair(
        os.path.join(emnist_path,'emnist-letters-test-images-idx3-ubyte'),
        os.path.join(emnist_path,'emnist-letters-test-labels-idx1-ubyte'),
        width
    )

    return training_x, training_y, test_x, test_y

def load_data_pair(images_path, labels_path, width):
    X, Y = loadlocal_mnist(images_path, labels_path)

    ucf_chars = [
        ord('u')-ord('a'),
        ord('c')-ord('a'),
        ord('f')-ord('a')
    ]

    indices = np.argwhere(Y == 15)
    indices = np.append(indices, np.argwhere(Y == 24))

    np.random.shuffle(indices)

    X_ucf_list = [(zoom(arr.reshape(28, 28), (width/28.0))).flatten() for arr in X[indices]]
    X_ucf = np.array(X_ucf_list)

    Y_ucf = Y[indices]

    one_hot_lookup = {
        15 : np.array([1, 0]),
        24 : np.array([0, 1])
        }

    # one_hot_lookup = {
    #     1 : np.array([1, 0, 0]),
    #     2 : np.array([0, 1, 0]),
    #     3 : np.array([0, 0, 1])
    #     }

    Y_ucf_one_hot = np.array([one_hot_lookup[n] for n in Y_ucf])

    return X_ucf, Y_ucf_one_hot


def main():
    width = 10

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
