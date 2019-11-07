import itertools
import matplotlib.pyplot as plt

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
        for output in output_lists:
            plt.plot(input_list, output)

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
        output_count = sum(self.neurons_per_layer)

        with SMBus(1) as bus:
            bus.write_byte(self.mcu_addr, self.commands["read_outputs"])
            
            outputs = [bus.read_byte_data(self.mcu_addr, self.commands["read_outputs"]) for x in range(output_count)]

        return outputs

    def set_weights_i2c(self, weights):
        # Flatten the nested lists of weights.
        data = itertools.chain(*(itertools.chain(*weights)))

        cmd_msg  = i2c_msg.write(self.mcu_addr, [self.commands['set_weights']]+[x for x in data])
        #data_msg = i2c_msg.write(self.mcu_addr, data)
        
        with SMBus(1) as bus:
            bus.i2c_rdwr(cmd_msg)
            #bus.i2c_rdwr(data_msg)
            
        #print(f'Sending {weights}')

    def read_outputs_i2c(self):
        output_count = sum(self.neurons_per_layer)

        write = i2c_msg.write(self.mcu_addr, [1])
        read  = i2c_msg.read(self.mcu_addr, output_count*2)

        with SMBus(1) as bus:
            bus.i2c_rdwr(read)
            
        #print(f'\nReceived output byte array: {list(read.buf[0:output_count*2])}\n')
        return [int.from_bytes(read.buf[i*2:i*2+2], byteorder='little') for i in range(output_count)]
    
def main():
    link = MLPLink(4, 2, [4,1], [4,4])
    weights = [ [ [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0] ], [ [0,0,0,0] ] ]

    neuron_names = []
    for i in range(link.layer_count):
        for j in range(link.neurons_per_layer[i]):
            neuron_names.append(f'Neuron {i}_{j}')

    output_lists = [[] for i in range(sum(link.neurons_per_layer))]

    start_idx = 0
    for i in range(link.layer_count):
        end_idx = start_idx + link.neurons_per_layer[i]
        for synapse in range(link.inputs_per_layer[i]):
            for weight in range(0,256):
    
                #weights[i][0:][synapse] = weight
                for w in weights[i]:
                    w[synapse] = weight
#                print(weights)
                link.set_weights_i2c(weights)

                outputs = link.read_outputs_i2c()

                for j in range(start_idx, end_idx):
                    output_lists[j].append(outputs[j])
        start_idx = end_idx
   
    link.plot_activation([x for x in range(256*4)], output_lists, neuron_names)

if __name__ == '__main__':
    main()
