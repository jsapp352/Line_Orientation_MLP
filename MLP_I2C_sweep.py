import itertools
import matplotlib.pyplot as plt

from I2CMLP import MLPLink
#from smbus2 import SMBus, i2c_msg
from time import sleep

def plot_activation(input_list, output_lists, neuron_names):
    for output in output_lists:
        plt.plot(input_list, output)

    plt.title('Activation function output vs. weight (all inputs at max value)')
    plt.xlabel('Byte value of weight')
    plt.ylabel('ADC output reading')
    plt.legend(neuron_names)

    plt.show()

    
def main():
    # link = MLPLink([4,3], [4,4])    
    # weights = [ [ [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0] ], [ [0,0,0,0], [0,0,0,0], [0,0,0,0]  ] ]

    link = MLPLink([3,3], [100,3])
    weights = [ [[0] * 100 ] * 4, [[0] * 3 ] * 3 ]

    inputs = [255] * 10 + [0] * 90

    link.set_inputs(inputs)

    neuron_names = []
    for i in range(link.layer_count):
        for j in range(link.neurons_per_layer[i]):
            neuron_names.append(f'Neuron {i}_{j}')

    output_lists = [[] for i in range(sum(link.neurons_per_layer))]
    output_count = 0

    start_idx = 0
    for i in range(link.layer_count):
        end_idx = start_idx + link.neurons_per_layer[i]
        for synapse in range(link.inputs_per_layer[i]):
            for weight in range(0,256,32):
    
                #weights[i][0:][synapse] = weight
                for w in weights[i]:
                    w[synapse] = weight
#                print(weights)
                link.set_weights(weights)
                
                outputs = link.read_outputs()

                for j in range(start_idx, end_idx):
                    output_lists[j].append(outputs[j])
                    output_count += 1
        start_idx = end_idx
   
    plot_activation([x for x in range(800)], [output_lists[0]], neuron_names)

if __name__ == '__main__':
    main()
