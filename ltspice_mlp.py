# A module that provides an interface between a software MLP model
#  and an LTspice hardware simulation.

import numpy
from pprint import pprint

_v_plus = 5
_v_minus = -5

_v_in_pos = 1.5
_v_in_neg = -1.5

class MLP_Circuit():
    def __init__(self, neural_network):
        self.neural_network = neural_network

        self.v_plus = _v_plus
        self.v_minus = _v_minus

        self.v_in_pos = _v_in_pos
        self.v_in_neg = _v_in_neg

        self.layer_inputs = []
        self.layer_outputs = []


    def create_layer_subcircuit(self, neuron_layer, inputs, layer_number, max_weight, r_total_ohms):
        # Set 'potentiometer' resistor values to represent synaptic weights
        synapses_r_pos = (1 + (neuron_layer.synaptic_weights / max_weight)) / 2 * r_total_ohms
        synapses_r_neg = r_total_ohms - synapses_r_pos

        #DEBUG
        # pprint(synapses_r_pos)
        # pprint(synapses_r_neg)


        outputs = []

        lines = []

        for i in range(0, neuron_layer.neuron_count):
            output = f'Neuron_{layer_number}_{i}_out'
            outputs.append(output)

            lines.append(f'XV3_{layer_number}_{i} 0 N001_{layer_number}_{i} V+ V- {output} TL084 TL084')
            lines.append(f'XV2_{layer_number}_{i} Neuron_{layer_number}_{i}_008 Neuron_{layer_number}_{i}_004 V+ V- N005 TL084')
            lines.append(f'R**_{layer_number}_{i} Neuron_{layer_number}_{i}_005 Neuron_{layer_number}_{i}_004 100k tol=5')
            lines.append(f'R4_{layer_number}_{i} Neuron_{layer_number}_{i}_001 Neuron_{layer_number}_{i}_005 20k tol=5')
            lines.append(f'R6_{layer_number}_{i} Input_{layer_number}_{i}_008 Neuron_{layer_number}_{i}_007 5k tol=5')
            lines.append(f'R8_{layer_number}_{i} Neuron_{layer_number}_{i}_003 Neuron_{layer_number}_{i}_001 100k tol=5')
            lines.append(f'R9_{layer_number}_{i} Neuron_{layer_number}_{i}_003 Neuron_{layer_number}_{i}_out 3.3k tol=5')
            lines.append(f'R12_{layer_number}_{i} Neuron_{layer_number}_{i}_004 Neuron_{layer_number}_{i}_ 10k tol=5')
            lines.append(f'D1_{layer_number}_{i} Neuron_{layer_number}_{i}_001 Neuron_{layer_number}_{i}_002 1N4001')
            lines.append(f'D2_{layer_number}_{i} Neuron_{layer_number}_{i}_002 Neuron_{layer_number}_{i}_003 1N4001')
            lines.append(f'D3_{layer_number}_{i} Neuron_{layer_number}_{i}_006 Neuron_{layer_number}_{i}_001 1N4001')
            lines.append(f'D4_{layer_number}_{i} Neuron_{layer_number}_{i}_003 Neuron_{layer_number}_{i}_006 1N4001')
            lines.append('\n')

            for j in range(0, neuron_layer.inputs_per_neuron):
                input = inputs[j]

                lines.append(f'XV_buff_{layer_number}_{i}_{j} {input} buff_out_{layer_number}_{i}_{j} V+ V- buff_out_{layer_number}_{i}_{j} TL084')
                lines.append(f'XV_buff`_{layer_number}_{i}_{j} 0 buff`_inv_{layer_number}_{i}_{j} V+ V- buff`_out_{layer_number}_{i}_{j} TL084')
                lines.append(f'R1_buff`_{layer_number}_{i}_{j} {input} buff`_inv_{layer_number}_{i}_{j} 100k')
                lines.append(f'R2_buff`_{layer_number}_{i}_{j} buff`_inv_{layer_number}_{i}_{j} buff_out_{layer_number}_{i}_{j} 100k')

                lines.append(f'R_in_{layer_number}_{i}_{j} buff_out_{layer_number}_{i}_{j} Input_{layer_number}_{i}_{j} {synapses_r_pos[j][i]}')
                lines.append(f'R_in`_{layer_number}_{i}_{j} buff`_out_{layer_number}_{i}_{j} Input_{layer_number}_{i}_{j} {synapses_r_neg[j][i]}')
                lines.append(f'R_in_{layer_number}_{i}_{j} Input_{layer_number}_{i}_{j} Input_{layer_number}_{i}')
                lines.append('\n')


        #DEBUG
        pprint(lines)
