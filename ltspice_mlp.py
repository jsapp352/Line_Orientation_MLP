# A module that provides an interface between a software MLP model
#  and an LTspice hardware simulation.

import numpy
from pprint import pprint

_v_plus = 5
_v_minus = -5

_v_in_pos = 1.5
_v_in_neg = -1.5

_pot_tolerance = 0.20

_r_total_ohms = 5000

_output_filename = 'MLP_netlist.net'

class MLP_Circuit_Layer():
    def __init__(self, neuron_layer, layer_number, r_total_ohms):
        self.neuron_layer = neuron_layer
        self.neuron_count = neuron_layer.neuron_count
        self.inputs_per_neuron = neuron_layer.inputs_per_neuron

        self.input_nodes = []
        self.output_nodes = []

        self.layer_number = layer_number
        self.r_total_ohms = r_total_ohms
        self.max_weight = self.neuron_layer.max_weight

        self.synapses_r_pos = None
        self.synapses_r_neg = None
        self.update_synapse_weights()

    def update_synapse_weights(self):
        #DEV -- Need to add method for randomizing actual resistance/representing pot. tolerance.
        self.synapses_r_pos = (1 + (self.neuron_layer.synaptic_weights / self.max_weight)) / 2 * self.r_total_ohms
        self.synapses_r_neg = self.r_total_ohms - self.synapses_r_pos
    
    def create_layer_subcircuit(self):
        inputs = []
        outputs = []

        neuron_lines = []

        synapse_lines = []

        for i in range(0, self.neuron_count):
            n_id = f'{self.layer_number}_{i}'
            output = f'Neuron_{n_id}_out'
            outputs.append(output)

            neuron_lines += self.create_neuron_subcircuit(n_id, output)

            for j in range(0, self.inputs_per_neuron):
                s_id = f'{n_id}_{j}'
                r_pos = f'{self.synapses_r_pos[j][i]}'
                r_neg = f'{self.synapses_r_neg[j][i]}'
                
                input = f'Synapse_{s_id}_in'
                inputs.append(input)
                
                synapse_lines += self.create_synapse_subcircuit(n_id, s_id, input, r_pos, r_neg)                

        self.output_nodes = outputs

        lines = neuron_lines + synapse_lines

        return lines

    def create_neuron_subcircuit(self, n_id, output):
        lines = []

        lines.append(f'XV3_{n_id} 0 N001_{n_id} V+ V- {output} TL084 TL084')
        lines.append(f'XV2_{n_id} Neuron_{n_id}_008 Neuron_{n_id}_004 V+ V- N005 TL084')
        lines.append(f'R**_{n_id} Neuron_{n_id}_005 Neuron_{n_id}_004 100k tol=5')
        lines.append(f'R4_{n_id} Neuron_{n_id}_001 Neuron_{n_id}_005 20k tol=5')
        lines.append(f'R6_{n_id} Input_{n_id}_008 Neuron_{n_id}_007 5k tol=5')
        lines.append(f'R8_{n_id} Neuron_{n_id}_003 Neuron_{n_id}_001 100k tol=5')
        lines.append(f'R9_{n_id} Neuron_{n_id}_003 Neuron_{n_id}_out 3.3k tol=5')
        lines.append(f'R12_{n_id} Neuron_{n_id}_004 Neuron_{n_id}_ 10k tol=5')
        lines.append(f'D1_{n_id} Neuron_{n_id}_001 Neuron_{n_id}_002 1N4001')
        lines.append(f'D2_{n_id} Neuron_{n_id}_002 Neuron_{n_id}_003 1N4001')
        lines.append(f'D3_{n_id} Neuron_{n_id}_006 Neuron_{n_id}_001 1N4001')
        lines.append(f'D4_{n_id} Neuron_{n_id}_003 Neuron_{n_id}_006 1N4001')
        lines.append('\n')

        return lines

    def create_synapse_subcircuit(self, n_id, s_id, input, r_pos, r_neg):
        lines  = []

        lines.append(f'XV_buff_{s_id} {input} buff_out_{s_id} V+ V- buff_out_{s_id} TL084')
        lines.append(f'XV_buff`_{s_id} 0 buff`_inv_{s_id} V+ V- buff`_out_{s_id} TL084')
        lines.append(f'R1_buff`_{s_id} {input} buff`_inv_{s_id} 100k')
        lines.append(f'R2_buff`_{s_id} buff`_inv_{s_id} buff_out_{s_id} 100k')

        lines.append(f'R_in_{s_id} buff_out_{s_id} Input_{s_id} {r_pos}')
        lines.append(f'R_in`_{s_id} buff`_out_{s_id} Input_{s_id} {r_neg}')
        lines.append(f'R_in_{s_id} Input_{s_id} Input_{n_id}')
        lines.append('\n')

        return lines

class MLP_Circuit():
    def __init__(self, neural_network):
        self.v_plus = _v_plus
        self.v_minus = _v_minus
        
        self.v_in_pos = _v_in_pos
        self.v_in_neg = _v_in_neg
        
        self.r_total_ohms = _r_total_ohms
        
        self.output_filename = _output_filename
        
        self.neural_network = neural_network
        self.hardware_layers = []
        self.initialize_hardware_layers()
        
    
    def initialize_hardware_layers(self):
        for idx,model_layer in enumerate(self.neural_network.neuron_layers):
            self.hardware_layers.append(MLP_Circuit_Layer(model_layer, idx, self.r_total_ohms))
    
    def create_netlist(self):
        lines = []

        for layer in self.hardware_layers:
            lines += layer.create_layer_subcircuit()

        #DEBUG
        pprint(lines)

        with open(self.output_filename, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')


    

    