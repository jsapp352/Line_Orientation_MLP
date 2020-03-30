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

    plt.title('Activation function output vs. weight (all inputs at max value)')
    plt.xlabel('Byte value of weight')
    plt.ylabel('ADC output reading')
    plt.legend(neuron_names)

    plt.show()
    
def main():
    link = link = MLPLink([3,3], [100,3])
#    link = MLPLink(4, 2, [4,1], [4,4])

    neuron_names = []
    for i in range(link.layer_count):
        for j in range(link.neurons_per_layer[i]):
            neuron_names.append(f'Neuron {i}_{j}')

    output_lists = [[] for i in range(sum(link.neurons_per_layer))]

    inputs = [0 <= x < 40 for x in range(100)]

    link.set_inputs(inputs)


    reps_count = 1

    for h in range(reps_count):
        start_idx = 0
        for i in range(0, 256):
            weights = [[[i] * 100] * 3, [[i] * 3] * 3]

            link.set_weights(weights)

            outputs = link.read_outputs()

            for i in range(len(outputs)):
                output_lists[i].append(outputs[i])


            


        # start_idx = 0
        # weights = [ [ [[0] * 100] * 3  ], [ [0,0,0] * 3 ] ]
        # for i in range(link.layer_count):
        #     end_idx = start_idx + link.neurons_per_layer[i]
        #     input_val = 0

        #     for synapse in range(link.inputs_per_layer[i]):
        #         for weight in range(0,256):
    
        #             #weights[i][0:][synapse] = weight
        #             for w in weights[i]:
        #                 w[synapse] = weight
        #                 #print(weights)
                    
        #             link.set_weights(weights)

        #             outputs = link.read_outputs()

        #             for j in range(start_idx, end_idx):
        #                 output_lists[j].append(outputs[j])
        #     start_idx = end_idx
   
    base_input_list = [x for x in range(256)]
    input_list = []
    for i in range(reps_count):
        input_list += base_input_list

    #output_lists.append([1/(1+exp(-(x-512)/10.28))*8192 for x in input_list])
    #neuron_names.append('Ideal sigmoid')

    plot_activation(input_list, output_lists, neuron_names)

if __name__ == '__main__':
    main()
