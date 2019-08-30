# A module that provides an interface between a software MLP model
#  and an LTspice hardware simulation.

import numpy
import re
import subprocess

from pprint import pprint

_v_plus = 5
_v_minus = -5

_v_in_max = 1.5
_v_in_min = -1.5

_pot_tolerance = 0.20

_r_total_ohms = 5000

_output_filename = 'MLP_netlist.net'
_result_filename = _output_filename[:-4] + '.log'
_ltspice_path = '"C:\\Program Files\\LTC\\LTspiceXVII\\XVIIx64.exe" -b '

_simulation_includes_input_buffers = False

class MLP_Circuit_Layer():
    def __init__(self, neuron_layer, layer_number, r_total_ohms):
        self.neuron_layer = neuron_layer
        self.neuron_count = neuron_layer.neuron_count
        self.inputs_per_neuron = neuron_layer.inputs_per_neuron

        self.input_nodes = [f'Synapse_{layer_number}_{x}_in' for x in range(0, self.inputs_per_neuron)] if layer_number == 0 else []
        self.output_nodes = []

        self.layer_number = layer_number
        self.r_total_ohms = r_total_ohms
        self.max_weight = self.neuron_layer.max_weight

        self.synapses_r_pos = None
        self.synapses_r_neg = None
        self.update_synapse_weights()

    def update_synapse_weights(self):
        #DEV -- Need to add method for randomizing actual resistance/representing pot. tolerance.
        max_val = numpy.amax(self.neuron_layer.synaptic_weights)
        # print(f'max_val: {max_val}')
        min_val = numpy.amin(self.neuron_layer.synaptic_weights)
        # print(f'min_val: {min_val}')

        # Scale weights to take up maximum pot. range.
        scale = 1 / (max(abs(max_val), abs(min_val)))
        # scale = 1 / self.max_weight

        # Scale weights down a little bit.
        scale *= 0.6

        self.synapses_r_pos = (1 + self.neuron_layer.synaptic_weights * scale) / 2 * self.r_total_ohms

        # self.synapses_r_pos = (1 + (self.neuron_layer.synaptic_weights / self.max_weight)) / 2 * self.r_total_ohms

        self.synapses_r_neg = self.r_total_ohms - self.synapses_r_pos
    
    def create_layer_subcircuit(self):
        outputs = []
        neuron_lines = []
        synapse_lines = []
        
        inputs = self.input_nodes
        
        for i in range(0, self.neuron_count):
            n_id = f'{self.layer_number}_{i}'
            output = f'Neuron_{n_id}_out'
            outputs.append(output)

            neuron_lines += self.create_neuron_subcircuit(n_id, output)

            for j in range(0, self.inputs_per_neuron):
                s_id = f'{n_id}_{j}'
                r_pos = f'{self.synapses_r_pos[j][i]}'
                r_neg = f'{self.synapses_r_neg[j][i]}'
                
                input = inputs[j]
                
                synapse_lines += self.create_synapse_subcircuit(n_id, s_id, input, r_pos, r_neg)                

        self.output_nodes = outputs

        lines = neuron_lines + synapse_lines

        return lines

    def create_neuron_subcircuit(self, n_id, output):
        lines = []

        lines.append(f'X_Sum_{n_id} Neuron_{n_id}_008 Neuron_{n_id}_004 V+ V- Neuron_{n_id}_005 TL084')
        lines.append(f'X_Activation_{n_id} 0 Neuron_{n_id}_001 V+ V- {output} TL084')
        lines.append(f'R**_{n_id} Neuron_{n_id}_005 Neuron_{n_id}_004 50k tol=5')
        lines.append(f'R4_{n_id} Neuron_{n_id}_001 Neuron_{n_id}_005 20k tol=5')
        lines.append(f'R8_{n_id} Neuron_{n_id}_003 Neuron_{n_id}_001 100k tol=5')
        lines.append(f'R9_{n_id} Neuron_{n_id}_003 {output} 3.3k tol=5')
        lines.append(f'R12_{n_id} Neuron_{n_id}_004 0 10k tol=5')
        lines.append(f'D1_{n_id} Neuron_{n_id}_001 Neuron_{n_id}_002 1N4001')
        lines.append(f'D2_{n_id} Neuron_{n_id}_002 Neuron_{n_id}_003 1N4001')
        lines.append(f'D3_{n_id} Neuron_{n_id}_006 Neuron_{n_id}_001 1N4001')
        lines.append(f'D4_{n_id} Neuron_{n_id}_003 Neuron_{n_id}_006 1N4001')
        lines.append('\n')

        return lines

    def create_synapse_subcircuit(self, n_id, s_id, input, r_pos, r_neg):
        simulation_includes_input_buffers = _simulation_includes_input_buffers

        lines  = []

        if simulation_includes_input_buffers:
            lines.append(f'XV_buff_{s_id} {input} buff_out_{s_id} V+ V- buff_out_{s_id} TL084')
            lines.append(f'XV_buff`_{s_id} 0 buff`_inv_{s_id} V+ V- buff`_out_{s_id} TL084')
            lines.append(f'R1_buff`_{s_id} {input} buff`_inv_{s_id} 100k')
            lines.append(f'R2_buff`_{s_id} buff`_inv_{s_id} buff_out_{s_id} 100k')

            lines.append(f'R_in_{s_id} buff_out_{s_id} Input_{s_id} {r_pos}')
            lines.append(f'R_in`_{s_id} buff`_out_{s_id} Input_{s_id} {r_neg}')
            lines.append(f'R_in_{s_id}_series Input_{s_id} Neuron_{n_id}_008 {self.r_total_ohms}')
        else:
            lines.append(f'R_in_{s_id} {input} Input_{s_id} {r_pos}')
            lines.append(f'R_in`_{s_id} {input}` Input_{s_id} {r_neg}')
            lines.append(f'R_in_{s_id}_series Input_{s_id} Neuron_{n_id}_008 {self.r_total_ohms}')

        lines.append('\n')

        return lines

class MLP_Circuit():
    def __init__(self, neural_network):
        self.v_plus = _v_plus
        self.v_minus = _v_minus
        
        self.v_in_max = _v_in_max
        self.v_in_min = _v_in_min
        
        self.r_total_ohms = _r_total_ohms
        
        self.output_filename = _output_filename
        self.result_filename = _result_filename
        self.ltspice_path = _ltspice_path
        
        self.neural_network = neural_network
        self.hardware_layers = []
        self.initialize_hardware_layers()

        self.power_sources = {}
        self.power_sources['V+'] = _v_plus
        self.power_sources['V-'] = _v_minus

        self.input_sources = []
    
    def initialize_hardware_layers(self):
        for idx,model_layer in enumerate(self.neural_network.neuron_layers):
            self.hardware_layers.append(MLP_Circuit_Layer(model_layer, idx, self.r_total_ohms))
        
        for i in range(1, len(self.hardware_layers)):
            self.hardware_layers[i].input_nodes = self.hardware_layers[i-1].output_nodes


    def update_input_sources(self, input_array):
        # print('Update_input_sources() input_array:')
        # pprint(input_array)
        # print('\n')

        inputs = input_array.tolist()

        # print('Update_input_sources() inputs.tolist():')
        # pprint(inputs)
        # print('\n')

        # pprint(inputs)

        min_input = min(inputs)
        max_input = max(inputs)

        input_range = abs(max_input - min_input)
        v_in_range = self.v_in_max - self.v_in_min
        map_scale_factor = v_in_range / input_range
        
        map_input = lambda x: (x - min_input) * map_scale_factor + self.v_in_min

        self.input_sources = [map_input(x) for x in inputs]

        # pprint(self.input_sources)

    def create_header(self):
        lines = []

        lines.append('.model D D')
        lines.append('.lib C:\\Users\\jsapp\\Documents\\LTspiceXVII\\lib\\cmp\\standard.dio')
        lines.append('.include TL084.txt')
        lines.append('.include 1N4001.txt')
        
        lines.append('\n')

        return lines

    def create_footer(self):
        lines = []
        
        lines.append('.tran 0 100p')
        lines.append('.options plotwinsize=0 trtol=7')

        # Skip N-R iterative solving
        lines.append('.options noopiter')

        # Skip gmin stepping
        lines.append('.options gminsteps=0')

        lines.append('.end')

        lines.append('\n')

        return lines
    
    def create_measurements(self):
        lines = []
        
        for layer in self.hardware_layers:
            for node in layer.output_nodes:
                lines.append(f'.save V({node})')

        for layer in self.hardware_layers:
            for node in layer.output_nodes:
                lines.append(f'.measure V{node} avg V({node})')
        
        lines.append('\n')

        return lines
    
    def create_source_definitions(self):
        simulation_includes_input_buffers = _simulation_includes_input_buffers

        lines = []
        i = 0

        for node in self.power_sources:
            lines.append(f'V{i} {node} 0 {self.power_sources[node]}V Rser=0.1')
            i += 1
        
        lines.append('\n')

        for idx,voltage in enumerate(self.input_sources):
            node = self.hardware_layers[0].input_nodes[idx]
            lines.append(f'V{i} {node} 0 {voltage} Rser=0.1')
            
            # If input buffers are not included in the simulation, add voltage sources for inverted input values
            if not simulation_includes_input_buffers:
                lines.append(f'V{i}` {node}` 0 {voltage * (-1)} Rser=0.1')
            i += 1

        lines.append('\n')

        return lines
    
    def create_netlist(self):
        lines = []

        lines += self.create_header()

        for layer in self.hardware_layers:
            lines += layer.create_layer_subcircuit()
        
        lines += self.create_source_definitions()
        lines += self.create_measurements()
        lines += self.create_footer()

        with open(self.output_filename, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')

    def run_simulation(self):
        command_string = self.ltspice_path + self.output_filename
        subprocess.check_call(command_string)

    def get_outputs(self):
        outputs = []

        with open(self.result_filename, 'r') as f:
            result = f.read()
            
            for layer in self.hardware_layers:
                output_voltages = []

                for node in layer.output_nodes:
                    pattern = re.compile(f'({node.lower()}\)\)\=)(-?\d+\.\d+)(\s)')
                    output_voltages.append(float(pattern.search(result)[2]))

                #DEV Break this normalization formula out into its own function
                outputs.append((numpy.asarray(output_voltages) + self.v_in_max) / (2 * self.v_in_max))
                # pprint(outputs[-1])

        # pprint(numpy.asarray(outputs))
        return numpy.asarray(outputs)

    def think(self, inputs):
        # print('Ckt.Think() inputs:')
        # pprint(inputs)
        # print('\n')
        # print('Ckt.Think() inputs.tolist():')
        # pprint(inputs.tolist())
        # print('\n')

        self.update_input_sources(inputs)

        for layer in self.hardware_layers:
            layer.update_synapse_weights()
        
        self.create_netlist()
        self.run_simulation()
        
        outputs = self.get_outputs()

        print(outputs)

        return outputs







    

    