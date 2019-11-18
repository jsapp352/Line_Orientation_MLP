# A module that provides an interface between a software MLP model
#  and digital potentiometer-based MLP hardware.

import I2CMLP
import numpy as np
from pprint import pprint

_pot_bit_resolution = 8
_pot_tap_count = 1 << _pot_bit_resolution

class MLP_Circuit():
    def __init__(self, neural_network):
        inputs_per_layer = [x.inputs_per_neuron for x in neural_network.neuron_layers]
        neurons_per_layer = [x.neuron_count for x in neural_network.neuron_layers]

        self.link = I2CMLP.MLPLink(neurons_per_layer, inputs_per_layer)

        self.neural_network = neural_network

        self.input_sources = []

    def calculate_layer_weights(self, layer):
        calculate_pos = lambda weight: min(_pot_tap_count-1, max(1, int((weight / layer.max_weight + 1) / 2 * _pot_tap_count)))

        return [calculate_pos(x) for x in layer.synaptic_weights]
    
    def update_synaptic_weights(self):

        weights = [self.calculate_layer_weights(x) for x in self.neural_network.neuron_layers]

        self.link.set_weights(weights)
    
    def update_input_values(self, input_array):

        inputs = input_array.tolist()

        self.link.set_inputs(inputs)

    def get_outputs(self):

        data = self.link.read_outputs()

        idx = 0        
        outputs = []

        for layer in self.neural_network:
            layer_outputs = []

            for _ in range(0, layer.neuron_count):
                layer_outputs.append(data[idx])
                idx += 1
            
            outputs.append((np.asarray(layer_outputs) / self.link.max_adc))
        
        return np.asarray(outputs)

    def think(self, inputs):

        self.update_synaptic_weights()

        self.update_input_values(inputs)

        outputs = self.get_outputs()

        print(outputs)

        return outputs







    

    
