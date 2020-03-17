import itertools
import numpy as np
from time import sleep

from smbus2 import SMBus, i2c_msg

_mcu_addr = 4
_adc_bit_resolution = 13

class MLPLink:
    def __init__(self, neurons_per_layer, inputs_per_layer):

        # Command codes for software/firmware communication. These should match the
        #   the definitions in the firmware code.
        self.commands = {
            "initialize": 1,
            "set_weights": 2,
            "set_inputs": 3,
            "read_outputs": 5
            }
        
        self.mcu_addr = _mcu_addr
        self.neurons_per_layer = neurons_per_layer
        self.inputs_per_layer = inputs_per_layer
        self.layer_count = len(neurons_per_layer)

        self.min_adc = 0.0
        self.max_adc = float(1 << _adc_bit_resolution) - 1.0

    def set_weights(self, weights):

        data = []

        for layer_weights in weights:
            for i in range(len(layer_weights[0])):
                for j in range(len(layer_weights)):
                    data.append(layer_weights[j][i])
            
        #print(f'Flattened weights: {data}')

        #print(f'Sending weights {data}\n')

        command = self.commands['set_weights']

        self.send_data(command, data)
    
    def set_inputs(self, inputs):

        #print(f'I2CMLP.set_inputs() inputs: {inputs}')

        packed_inputs = list(np.packbits(inputs))

        command = self.commands['set_inputs']

        self.send_data(command, packed_inputs)

    def read_outputs(self):

        max_adc = self.max_adc

        output_count = 6 #sum(self.neurons_per_layer)

        read  = i2c_msg.read(self.mcu_addr, output_count*2)

        # Initialize output list to make a do-while-type loop in the I2C read section.
        outputs = [max_adc + 1]

        success = False
        while not success:
            try:
                self.send_data(self.commands['read_outputs'], [])
                success = True
            except:
                success = False

        #DEBUG
        # return [0] * output_count

        sleep(0.01)

        with SMBus(1) as bus:
            success = False

            while not success or any([x > max_adc for x in outputs]):
                try:
                    bus.i2c_rdwr(read)
                    success = True
                except:
                    success = False
                    continue

                outputs = [int.from_bytes(read.buf[i*2:i*2+2], byteorder='little') for i in range(output_count)]
                print(*read.buf[0:12])

        sleep(0.01)

#        print(outputs)

        return [0] * 7  
        return outputs
    
    def send_data(self, command, payload):
        data = [command, (len(payload) & 0x00FF), ((len(payload) & 0xFF00) >> 8)]

        # print(data)

        msg = i2c_msg.write(self.mcu_addr, data)

        with SMBus(1) as bus:
            bus.i2c_rdwr(msg)
        
        sleep(.0005)

        if len(data) > 0:

            msg = i2c_msg.write(self.mcu_addr, payload)

            with SMBus(1) as bus:
                bus.i2c_rdwr(msg)
        
            sleep(0.01)


def main():
    link = MLPLink([3,3], [100,3])

    while True:
        link.read_outputs()

if __name__ == '__main__':
    main()