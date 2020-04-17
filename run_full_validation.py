import emnist_mlp
from emnist_mlp import NeuralNetwork, NeuronLayer
from numpy import array
from time import sleep
import sys
import timeit

def run_validation():
    global validation_set_inputs
    global validation_set_outputs
    global accuracy
    global min_difference

    accuracy, min_difference = neural_network.validate(validation_set_inputs, validation_set_outputs, input_indices)

def run_single_think():    
    global single_input

    outputs = neural_network.hardware_think(single_input, False)

filename = 'emnist_mlp_CAT_2020_04_12_20_49_43_97p9.pickle'

if len(sys.argv) > 1:
    filename = sys.argv[1]

neural_network = emnist_mlp.deserialize_neural_network(filename)

all_input_data = emnist_mlp.load_data(neural_network.data_char_set)

training_set_inputs, training_set_outputs, validation_set_inputs, validation_set_outputs = all_input_data

input_indices = [x for x in range(len(validation_set_inputs))]

single_input = array([validation_set_inputs[0]])

if neural_network.ckt != None:
    print('Setting hardware synapse weights.')
    print('')
    neural_network.ckt.update_synaptic_weights()
    
sleep(0.05)

print('')
print('Validating network\'s prediction accuracy...')

accuracy = None
min_difference = None
count = 1
runtime = timeit.timeit('run_validation()', globals=globals(), number=count) / count

print('Validation completed.')
print('')

sleep(0.05)

print('Running repeated single-image recognition calls to test latency...')

count = 10000
single_runtime = timeit.timeit('run_single_think()', globals=globals(), number=count) / count

print('Latency testing completed.')
print('')

if neural_network.ckt != None:
    neural_network.ckt.reset_network()

print('')
print(f'--------------------------------------------------------------------------------------')
print(f' Validation for saved network file \'{filename}\'')
print(f'--------------------------------------------------------------------------------------')
print('')
print(f' Dataset:    {neural_network.data_char_set}')
print('')
print(f' Runtime:    {len(validation_set_inputs)} recognition operations ran in {runtime:0.3f} seconds.')
print('')
print(f' Throughput: {(len(validation_set_inputs)/runtime):0.2f} operations per second')
print('')
print(f' Latency:    {single_runtime*1000:0.2f} ms (average from {count} single-image recognition calls)')
print('')
print(f' Accuracy:   {accuracy:0.2f}%')
print('')
print('')
print(f'--------------------------------------------------------------------------------------')
print('')
