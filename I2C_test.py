import numpy as np
from time import sleep

from smbus2 import SMBus, i2c_msg

_mcu_addr = 4
_adc_bit_resolution = 16

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

        success = False
        while not success:
            try:
                self.send_data(command, data)
                success = True
            except:
                # sleep(1)
                success = True
                pass
        
        sleep(0.01)
    
    def set_inputs(self, inputs):

        #print(f'I2CMLP.set_inputs() inputs: {inputs}')

        packed_inputs = list(np.packbits(inputs))

        command = self.commands['set_inputs']

        success = False
        while not success:
            try:
                self.send_data(command, packed_inputs)
                success = True
            except:
                pass
        
        sleep(0.01)

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

                with SMBus(1) as bus:
                    bus.i2c_rdwr(read)

                success = True

            except Exception as ex:
                # print(f'failed read_output command rx {ex}')
                
                sleep(0.1)
                success = False
                success = True

        outputs = [int.from_bytes(read.buf[i*2:i*2+2], byteorder='little') for i in range(output_count)]
        # print(*read.buf[0:12])
                
        print(outputs)

        sleep(0.01)


        return outputs
    
    def send_data(self, command, payload):
        data = [command, (len(payload) & 0x00FF), ((len(payload) & 0xFF00) >> 8)]

        print(data) 

        msg = i2c_msg.write(self.mcu_addr, data)

        with SMBus(1) as bus:
            bus.i2c_rdwr(msg)
        
        sleep(.0001)

        if len(payload) > 0:

            msg = i2c_msg.write(self.mcu_addr, payload)

            with SMBus(1) as bus:
                bus.i2c_rdwr(msg)
        
            sleep(0.0001)            


def main():
    link = MLPLink([3,3], [100,3])
    inputs = [255 if x<60 else 0 for x in range(100)] 
    while True:
        for i in range(0,256):
            link.set_inputs(inputs)
            weights = [[[i] * 100] * 3, [[255-i] * 3]*3]
            link.set_weights(weights)
            # sleep(0.5)

            link.read_outputs()

if __name__ == '__main__':
    main()