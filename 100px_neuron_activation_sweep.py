import itertools
import matplotlib.pyplot as plt
from I2CMLP import MLPLink
from numpy import tanh
from smbus2 import SMBus, i2c_msg
from time import sleep

def plot_activation(input_list, output_lists, ideal_outputs, neuron_names):
    neuron_names.append('Ideal activation function')

    for output_list in output_lists:
        plt.plot(input_list, output_list, '.', markersize=5)
    
    
    plt.plot(sorted(input_list), ideal_outputs, '-')

    # plt.plot(input_list[0:1024], ideal_sigmoid)

    plt.title('Daughterboard output vs. input activation (all weights at max value)')
    plt.xlabel('Input Activation (swept in successive groups)')
    plt.ylabel('ADC output reading')
    plt.legend(neuron_names)

    plt.show()
    
def main():
    link = link = MLPLink([3,3], [100,3])

    ideal_activation = lambda x: tanh((x - 0.0005) / 10) * 10

    daughterboard_names = []
    for i in range(1,4):
        daughterboard_names.append(f'Output neruon {i}')

    output_lists = [ [], [], [] ]
    ideal_outputs = []
    input_list = []

    reps_count = 1

    for h in range(reps_count):
        start_idx = 0
        active_input_width = 20

        while (start_idx + active_input_width <= 20):
            
            for i in range(active_input_width):
                inputs = [start_idx <= x < (start_idx + i) for x in range(100)]

                link.set_inputs(inputs)

                for j in range(31, 256, 32):
                    link.set_weights([ [[j, j, j]] * 100 , [[128,128,128],[128,128,128],[255,255,255]]])
                    link.set_inputs(inputs)

                    outputs = link.read_outputs()

                    daughterboard_sums = link.read_daughterboard_sums()
                    input_list.append((daughterboard_sums[2] - 32635) / 32636 / 3 / 2 * 1.65)

                    ideal_outputs.append(ideal_activation((daughterboard_sums[2]/65535 * 2 - 1.0) /3 /2) * 1.65)

                    output_lists[0].append((outputs[0] - 32635) / 32636 * 1.65)
                    output_lists[1].append((outputs[1] - 32635) / 32636 * 1.65)
                    output_lists[2].append((outputs[2] - 32635) / 32636 * 1.65)

                    print(f'db_sums: {daughterboard_sums}')


                    print(f'outputs: {outputs}')
                    # input_list.append((outputs[2] - 32635) / 32636 * 1.65)
                    # output_lists[0].append((outputs[3] - 32635) / 32636 * 1.65)
                    # output_lists[1].append((outputs[4] - 32635) / 32636 * 1.65)
                    # output_lists[2].append((outputs[5] - 32635) / 32636 * 1.65)

                
            start_idx += active_input_width


    link.set_inputs([0]*100)

    #output_lists.append([1/(1+exp(-(x-512)/10.28))*8192 for x in input_list])
    #neuron_names.append('Ideal sigmoid')

    plot_activation(input_list, output_lists, ideal_outputs, daughterboard_names)

if __name__ == '__main__':
    main()
