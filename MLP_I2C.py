import itertools
import smbus2

from time import sleep

class MLPLink:
    def __init__(self, mcu_addr, layer_count, neurons_per_layer, inputs_per_layer):
        self.commands = {
            "set_weights":  0,
            "read_outputs": 1,
            }
        
        self.mcu_addr = mcu_addr
        self.layer_count = layer_count
        self.neurons_per_layer = neurons_per_layer
        self.inputs_per_layer = inputs_per_layer

    def set_weights(self, weights):
        # Flatten the nested lists of weights.
        data = itertools.chain(*(itertools.chain(*weights)))
        
        with smbus2.SMBus(1) as bus:
            bus.write_byte(self.mcu_addr, self.commands["set_weights"])
            sleep(0.1)
            for weight in data:
                print(f'Sending {weight}')
                bus.write_byte(self.mcu_addr, weight)
    
    def read_outputs(self):
        output_count = sum(self.neurons_per_layer)

        with smbus2.SMBus(1) as bus:
            bus.write_byte(self.mcu_addr, self.commands["read_outputs"])
            
            outputs = [bus.read_byte(self.mcu_addr) for x in range(output_count)]

        return outputs
    
def main():
    link = MLPLink(4, 3, [2,2,2], [1,2,2])
    weights = [ [ [0],[2] ], [ [1,2],[3,4] ], [ [1,2],[3,4] ] ]

    link.set_weights(weights)

    outputs = link.read_outputs()

    print(outputs)



if __name__ == '__main__':
    main()
