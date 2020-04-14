import matplotlib.pyplot as plt
import emnist_mlp
from emnist_mlp import NeuralNetwork, NeuronLayer

def plot_comparison(files):
    networks = [emnist_mlp.deserialize_neural_network(x) for x in files]

    for network in networks:
        epoch, accuracy = network.accuracy_by_epoch
        plt.plot(epoch, accuracy)

    plt.title(f'Prediction Accuracy by Training Epoch')
    plt.legend([' '.join(x.data_char_set) for x in networks])

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.show()

def main():
    files = []

    files.append('emnist_mlp_UCF_2020_04_09_00_53_01_95p17.pickle')
    files.append('emnist_mlp_ABC_2020_04_09_01_31_47_92p78.pickle')
    files.append('emnist_mlp_XYZ_2020_04_09_01_53_52_85p0.pickle')
    files.append('emnist_mlp_BIG_2020_04_12_02_05_52_95p48.pickle')
    files.append('emnist_mlp_CAT_2020_04_12_01_21_40_97p44.pickle')

    plot_comparison(files)

if __name__ == '__main__':
    main()