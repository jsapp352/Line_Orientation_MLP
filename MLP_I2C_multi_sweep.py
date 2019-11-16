import itertools
import matplotlib.pyplot as plt

from numpy import exp
from smbus2 import SMBus, i2c_msg
from time import sleep

class MLPLink:
    def __init__(self, mcu_addr, layer_count, neurons_per_layer, inputs_per_layer):
        self.commands = {
            "set_weights":  0,
            "read_outputs": 5,
            }
        
        self.mcu_addr = mcu_addr
        self.layer_count = layer_count
        self.neurons_per_layer = neurons_per_layer
        self.inputs_per_layer = inputs_per_layer

    def plot_activation(self, input_list, output_lists, neuron_names):
        ideal_sigmoid = [1/(1+exp(-(x-512)/10.24))*8192 for x in input_list[0:1024]]
        neuron_names.append('Ideal sigmoid')

        for output in output_lists:
            plt.plot(input_list, output, '.', markersize=3)

        plt.plot(input_list[0:1024], ideal_sigmoid)

        plt.title('Activation function output vs. weight (all inputs at max value)')
        plt.xlabel('Byte value of weight')
        plt.ylabel('ADC output reading')
        plt.legend(neuron_names)

        plt.show()


    def set_all_weights(self, weight):
        weights = [[[weight if i==0 else 255 for k in range(self.inputs_per_layer[i])] for j in range(self.neurons_per_layer[i])] for i in range(self.layer_count)]
        
        self.set_weights_i2c(weights)

    def set_weights(self, weights):
        # Flatten the nested lists of weights.
        data = itertools.chain(*(itertools.chain(*weights)))
        
        with SMBus(1) as bus:
            bus.write_byte(self.mcu_addr, self.commands["set_weights"])
            sleep(0.1)
            for weight in data:
                print(f'Sending {weight}')
                bus.write_byte(self.mcu_addr, weight)
    
    def read_outputs(self):
        adc_resolution_bits = 13

        max_adc = 2 << adc_resolution_bits

        output_count = sum(self.neurons_per_layer)

        outputs = [max_adc + 1 for x in range(output_count)]

        while (outputs.any([x > max_adc for x in outputs])):
            with SMBus(1) as bus:
                bus.write_byte(self.mcu_addr, self.commands["read_outputs"])            
                outputs = [bus.read_byte_data(self.mcu_addr, self.commands["read_outputs"]) for x in range(output_count)]


        return outputs

    def set_weights_i2c(self, weights):
        # Flatten the nested lists of weights.
        data = list(itertools.chain(*(itertools.chain(*weights))))

        cmd_msg  = i2c_msg.write(self.mcu_addr, [self.commands['set_weights']]+data[::-1])
        #data_msg = i2c_msg.write(self.mcu_addr, data)
        
        with SMBus(1) as bus:
            bus.i2c_rdwr(cmd_msg)
            #bus.i2c_rdwr(data_msg)
            
        #print(f'Sending {weights}')

    def read_outputs_i2c(self):
        adc_resolution_bits = 13
        max_adc = 2 << adc_resolution_bits

        output_count = sum(self.neurons_per_layer)

        write = i2c_msg.write(self.mcu_addr, [1])
        read  = i2c_msg.read(self.mcu_addr, output_count*2)

        with SMBus(1) as bus:
            bus.i2c_rdwr(read)
            outputs = [int.from_bytes(read.buf[i*2:i*2+2], byteorder='little') for i in range(output_count)]

            while any([x > max_adc for x in outputs]):
                bus.i2c_rdwr(read)
                outputs = [int.from_bytes(read.buf[i*2:i*2+2], byteorder='little') for i in range(output_count)]

        return outputs
    
def main():
    link = MLPLink(4, 1, [4], [4])
#    link = MLPLink(4, 2, [4,1], [4,4])

    neuron_names = []
    for i in range(link.layer_count):
        for j in range(link.neurons_per_layer[i]):
            neuron_names.append(f'Neuron {i}_{j}')

    output_lists = [[] for i in range(sum(link.neurons_per_layer))]


    reps_count = 20

    for h in range(reps_count):
        start_idx = 0
        weights = [ [ [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0] ], [ [0,0,0,0] ] ]
        for i in range(link.layer_count):
            end_idx = start_idx + link.neurons_per_layer[i]
            input_val = 0

            for synapse in range(link.inputs_per_layer[i]):
                for weight in range(0,256):
    
                    #weights[i][0:][synapse] = weight
                    for w in weights[i]:
                        w[synapse] = weight
                        #print(weights)
                    
                    link.set_weights_i2c(weights)

                    outputs = link.read_outputs_i2c()

                    for j in range(start_idx, end_idx):
                        output_lists[j].append(outputs[j])
            start_idx = end_idx
   
    base_input_list = [x for x in range(256*4)]
    input_list = []
    for i in range(reps_count):
        input_list += base_input_list

    #output_lists.append([1/(1+exp(-(x-512)/10.28))*8192 for x in input_list])
    #neuron_names.append('Ideal sigmoid')

    link.plot_activation(input_list, output_lists, neuron_names)

if __name__ == '__main__':
    main()
