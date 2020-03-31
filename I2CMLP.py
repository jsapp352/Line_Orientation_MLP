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
            "read_outputs": 5,
            "read_db_sums": 6,
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
        
        sleep(0.005)
    
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
        
        sleep(0.0001)

    def read_outputs(self):

        max_adc = self.max_adc

        output_count = 6 #sum(self.neurons_per_layer)

        read  = i2c_msg.read(self.mcu_addr, output_count*2+1)

        success = False
        while not success:
            try:
                self.send_data(self.commands['read_outputs'], [])

                sleep(.0001)

                with SMBus(1) as bus:
                    bus.i2c_rdwr(read)

                success = True

            except Exception as ex:
                # print(f'failed read_output command rx {ex}')
                
                # sleep(0.001)
                # success = False
                success = True

        outputs = [int.from_bytes(read.buf[1:(output_count*2+1)][i*2:i*2+2], byteorder='little') for i in range(output_count)]
        # print(*read.buf[1:13])
                
        # print(outputs)

        # sleep(0.0001)


        return outputs

    def read_daughterboard_sums(self):

            max_adc = self.max_adc

            output_count = 3 #sum(self.neurons_per_layer)

            read  = i2c_msg.read(self.mcu_addr, output_count*2+1)
            
            try:
                self.send_data(self.commands['read_db_sums'], [])

                sleep(.01)

                with SMBus(1) as bus:
                    bus.i2c_rdwr(read)

            except Exception as ex:
                pass

            outputs = [int.from_bytes(read.buf[1:(output_count*2+1)][i*2:i*2+2], byteorder='little') for i in range(output_count)]
            # print(*read.buf[1:13])
                    
            # print(outputs)

            # sleep(0.0001)

            return outputs
        
    def send_data(self, command, payload):
        data = [command, (len(payload) & 0x00FF), ((len(payload) & 0xFF00) >> 8)]

        # print(data) 

        msg = i2c_msg.write(self.mcu_addr, data)

        with SMBus(1) as bus:
            bus.i2c_rdwr(msg)

        if len(payload) > 0:

            msg = i2c_msg.write(self.mcu_addr, payload)

            with SMBus(1) as bus:
                bus.i2c_rdwr(msg)
        
            # sleep(0.0001)