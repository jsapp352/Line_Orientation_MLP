import itertools
import numpy as np

from smbus2 import SMBus, i2c_msg

_mcu_addr = 0x04
_adc_bit_resolution = 13

class MLPLink:
    def __init__(self, neurons_per_layer, inputs_per_layer):

        # Command codes for software/firmware communication. These should match the
        #   the definitions in the firmware code.
        self.commands = {
            "initialize": 1,
            "set_weights": 2,
            "set_inputs": 3,
            "read_outputs": 4
            }
        
        self.mcu_addr = _mcu_addr
        self.neurons_per_layer = neurons_per_layer
        self.inputs_per_layer = inputs_per_layer
        self.layer_count = len(neurons_per_layer)

        self.min_adc = 0.0
        self.max_adc = float(1 << _adc_bit_resolution) - 1.0

    def set_weights(self, weights):

        # Flatten the nested lists of weights.
        data = list(itertools.chain(*(itertools.chain(*weights))))

        msg  = i2c_msg.write(self.mcu_addr, data.insert(0, self.commands['set_weights']))
        
        with SMBus(1) as bus:
            bus.i2c_rdwr(msg)
    
    def set_inputs(self, inputs):

        data = list(np.packbits(inputs))

        msg = i2c_msg.write(self.mcu_addr, data.insert(0, self.commands['set_inputs']))

        with SMBus(1) as bus:
            bus.i2c_rdwr(msg)

    def read_outputs(self):

        max_adc = self.max_adc

        output_count = sum(self.neurons_per_layer)

        read  = i2c_msg.read(self.mcu_addr, output_count*2)

        # Initialize output list to make a do-while-type loop in the I2C read section.
        outputs = [max_adc + 1]

        with SMBus(1) as bus:
            while any([x > max_adc for x in outputs]):
                bus.i2c_rdwr(read)
                outputs = [int.from_bytes(read.buf[i*2:i*2+2], byteorder='little') for i in range(output_count)]

        return outputs
