import itertools
import matplotlib.pyplot as plt
from I2CMLP import MLPLink

from numpy import exp
from smbus2 import SMBus, i2c_msg
from time import sleep

def plot_activation(input_list, output_lists, neuron_names):
    # ideal_sigmoid = [1/(1+exp(-(x-512)/10.24))* for x in input_list[0:1024]]
    # neuron_names.append('Ideal sigmoid')

    for output in output_lists:
        plt.plot(input_list, output, '.', markersize=5)

    # plt.plot(input_list[0:1024], ideal_sigmoid)

    plt.title('Daughterboard output vs. input activation (all weights at max value)')
    plt.xlabel('Input Activation (swept in successive groups)')
    plt.ylabel('ADC output reading')
    plt.legend(neuron_names)

    plt.show()
    
def main():
    link = link = MLPLink([3,3], [100,3])

    daughterboard_names = []
    for i in range(1,4):
        daughterboard_names.append(f'Daughterboard {i}')

    output_lists = [[], [], []]

    weights = [[[255] * 100] * 3, [[127] * 3] * 3]

    link.set_weights(weights)

    reps_count = 1

    for h in range(reps_count):
        start_idx = 0
        active_input_width = 50
        while (start_idx + active_input_width <= 100):
            
            for i in range(active_input_width):
                
                link.set_weights(weights)
                inputs = [start_idx <= x < (start_idx + i) for x in range(100)]

                link.set_inputs(inputs)

                daughterboard_sums = link.read_daughterboard_sums()

                print(daughterboard_sums)

                for j in range(len(daughterboard_sums)):
                    output_lists[j].append(daughterboard_sums[j])
                
            start_idx += active_input_width

   
    
    input_list = [x for x in range(1,101)]

    #output_lists.append([1/(1+exp(-(x-512)/10.28))*8192 for x in input_list])
    #neuron_names.append('Ideal sigmoid')

    plot_activation(input_list, output_lists[0:3], daughterboard_names)

if __name__ == '__main__':
    main()
