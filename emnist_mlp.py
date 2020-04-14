# Decription: A neural network model for the recognition of handrwritten characters.
# Author:     Justin Sapp
# Date:       Spring 2020

# Copyright (c) 2020 Justin Sapp

# Adapted from code by Milo Spencer-Harper. Source: https://github.com/miloharper/multi-layer-neural-network .
# Portions of this code were used under the terms of the license/copyright notice listed below:

# The MIT License (MIT)

# Copyright (c) 2015 Milo Spencer-Harper

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the "Software"), to deal in the Software without restriction, including without 
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of 
# the Software, and to permit persons to whom the Software is furnished to do so, subject to the following 
# conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial 
# portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
from datetime import datetime
import emnist_loader
from hardwareMLP import MLP_Circuit
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
# import handwritten_samples
from numpy import exp, array, random, dot, argmax
from pprint import pprint

_validation_iterations = 5181
_validation_tick_interval = 1

_learning_rate = 0.005
# _learning_rate = 0.005
_second_learning_rate = 0.001
_max_weight = 10.0

_minimum_accuracy = 95.17
_minimum_output_difference = 0.0

_starting_seed = 3
_seed_iterations = 1

_data_char_set = ['U', 'C', 'F']

_emnist_path = os.path.join(os.getcwd(), 'emnist_data')

# Standard deviation for activation noise
_standard_deviation = 0.15

_saved_network_path = os.path.join(os.getcwd(), 'saved_emnist_mlp_networks')

_use_software_model = False

_use_hardware_validation = False

_hardware_faults = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a multi-layer perceptron to detect handwritten characters.',
        allow_abbrev=True)

    parser.add_argument('epochs', type=int,
                        help='number of training epochs')
    parser.add_argument('training_batch_size', type=int, nargs='?', default=10000,
                        help='specify a maximum data-set size for training batch')
    parser.add_argument('-plot', action='store_true',
                        help='show a plot of the accuracy data by epoch')
    parser.add_argument('-noisy_activation', action='store_true',
                        help='add simulated noise to the activation function')
    parser.add_argument('-software_model', action='store_true',
                    help='use a software-based neural network model (no SPICE simulation)')
    parser.add_argument('-hardware_validation', action='store_true',
                        help='use the hardware neural network for validation, regardless of model used for training')
    parser.add_argument('-faults', action='store_true',
                        help='simulate hardware faults')

    _args = parser.parse_args()

    _use_software_model = _args.software_model

    _use_hardware_validation = _args.hardware_validation

    if(_args.faults):
        _hardware_faults = True
        _fault_rate = 0.05


def serialize_neural_network(neural_network, final_accuracy):
    datachars = ''.join(data_char_set)
    accuracy_string = f'{int(final_accuracy)}p{int((final_accuracy - int(final_accuracy))*100)}'
    software_label = 'software_' if _use_software_model else ''
    filename = os.path.join(_saved_network_path, f'emnist_mlp_{datachars}_{software_label}{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}_{accuracy_string}.pickle')

    with open(filename, 'wb') as f:
        pickle.dump(neural_network, f, pickle.HIGHEST_PROTOCOL)

def deserialize_neural_network(filename):
    neural_network = None

    with open(os.path.join(_saved_network_path, filename), 'rb') as f:
        neural_network = pickle.load(f)
    
    return neural_network

def plot_data_samples(X_train, y_labels, y_train, width, samples_per_row):
    fig = plt.figure()
    for i in range(len(X_train)):
        two_d = (np.reshape(X_train[i], (width, width)) * 255).astype(np.uint8)
        two_d = np.rot90(two_d, 3)
        two_d = np.fliplr(two_d)
        plt.subplot(samples_per_row,len(X_train)/samples_per_row,i+1)
        plt.tight_layout()
        plt.imshow(two_d, cmap='gray', interpolation='none')

        if y_labels != None:
            title_color = 'black' if y_labels[i] == y_train[i] else 'red'
            title_weight = 'normal'if y_labels[i] == y_train[i] else 'bold'
            plt.title(f"{y_labels[i]} -> {y_train[i]}", c=title_color, fontweight=title_weight)
        else:
            plt.title(f"{y_train[i]}", c='black', fontweight='normal')
        
        plt.xticks([])
        plt.yticks([])
    plt.show()

def plot_accuracy(accuracy_by_epoch):
    char_set = _data_char_set
    epoch, accuracy = accuracy_by_epoch
    plt.plot(epoch, accuracy)

    noise_label = f' (Activation noise standard dev.: {_standard_deviation})' if _args.noisy_activation else ''
    batch_size_label = f'Training batch size: {_args.training_batch_size}'
    plt.title(f'Prediction Accuracy by Training Epoch\nData set: {char_set}\n{batch_size_label}')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.show()

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.neuron_count = number_of_neurons
        self.inputs_per_neuron = number_of_inputs_per_neuron
        self.max_weight = _max_weight

        self.set_random_starting_weights()



    def set_random_starting_weights(self):
        self.synaptic_weights = random.normal(size=(self.inputs_per_neuron, self.neuron_count), scale=(self.inputs_per_neuron**-0.5))

    def adjust_weights(self, adjustments):
        max_weight = self.max_weight

        self.synaptic_weights += adjustments * _learning_rate
        abs_weights = np.abs(self.synaptic_weights)
        if (abs_weights > max_weight).any():
            self.synaptic_weights *= _max_weight / (abs_weights).max()

class NeuralNetwork():
    def __init__(self, neuron_layers):
        self.neuron_layers = neuron_layers

        self.activation_function = self.tanh
        self.activation_derivative = self.tanh_derivative

        self.data_char_set = _data_char_set
        
        self.ckt = MLP_Circuit(self)

        self.think = self.software_think if _use_software_model else self.hardware_think
        self.accuracy_by_epoch = None
        self.final_accuracy = None

    def tanh(self, x):
        return np.tanh((x - 0.0005) / 10 )
    
    def tanh_derivative(self, x):
        # return 1.0 - np.tanh(x)**2
        return 1 / (5 * (np.cosh(0.0001 - 0.2 * x) + 1))

    def relu_derivative(self, x):
        return x > 30
    
    def sigmoid(self, x):
        if _args.noisy_activation == True:
            x += random.normal(0, _standard_deviation, x.shape)

        return 1 / (1 + exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, validation_set_inputs, validation_set_outputs, number_of_training_iterations, minimum_accuracy, minimum_output_difference):
        batch_size = _args.training_batch_size
        layers = self.neuron_layers

        accuracy_by_epoch = ([], [])

        validation_iterations = _validation_iterations
        validation_tick_interval = _validation_tick_interval
        # validation_data_indices = [random.randint(0, len(validation_set_inputs)) for x in range(0, validation_iterations)]
        validation_data_indices = [x for x in range(0, min(validation_iterations, len(validation_set_inputs)))]

        last_layer_idx = len(layers) - 1

        def create_batches(ls):
            batches = []
            batch_range_end = int(len(ls) / batch_size) + 1
            for i in range(batch_range_end):
                batch = []
                
                # print(f'range({i*batch_size}, {min((i+1)*batch_size, len(ls))})')
                for j in range(i*batch_size, min((i+1)*batch_size, len(ls))):
                    # print(ls[j])
                    batch.append(ls[j])
                # print(f'ls[{i*batch_size} : {i*(batch_size+1)}]: {ls[i*batch_size : i*(batch_size+1)]}')
                batches.append(array(batch))
                # print(f'len(batches[{i}]): {len(batches[i])}')

            
            return array(batches)

        for epoch in range(number_of_training_iterations):
            # Shuffle the training set
            data_set_size = training_set_inputs.shape[0]
            indices = np.random.permutation(data_set_size)

            batch_indices = create_batches(indices)


            for batch_number in range(len(batch_indices)):
                # Pass the training set through the neural network.
                training_inputs = training_set_inputs[batch_indices[batch_number]]
                training_outputs = training_set_outputs[batch_indices[batch_number]]

                # print(training_inputs)
                # print(training_outputs)

                output_from_layers = self.think(training_inputs)

                deltas = {}
                errors = {}

                # We have to calculate the last layer's error value first.
                last_layer = layers[last_layer_idx]
                last_layer_output = output_from_layers[last_layer_idx]
                errors[last_layer] = training_outputs - last_layer_output
                deltas[last_layer] = errors[last_layer] * self.activation_derivative(last_layer_output)

                # Then we can loop through the rest of the layers in descending order
                #   and calculate their values.
                for i in range(last_layer_idx - 1, -1, -1):
                    layer = layers[i]
                    output = output_from_layers[i]

                    next_layer = layers[i+1]

                    errors[layer] = deltas[next_layer].dot(next_layer.synaptic_weights.T)
                    deltas[layer] = errors[layer] * self.activation_derivative(output)

                # Calculate how much to adjust the weights by
                adjustments = {}

                adjustments[layers[0]] = training_inputs.T.dot(deltas[layers[0]])

                for i in range (1, len(layers)):
                    adjustments[layers[i]] = output_from_layers[i-1].T.dot(deltas[layers[i]])

                # Adjust the weights.
                for layer in layers:
                    layer.adjust_weights(adjustments[layer])
                
                # if neural_network.fault_indices != None:
                #     neural_network.weights[fault_indices] = -_max_weight
                #     print(weights)
                #     input()
                # else:
                #     print('wha happen?')
                #     print(f'neural_network.fault_indices: {neural_network.fault_indices}')
                #     input()

            # Validate results
            if epoch % validation_tick_interval == 0:
                accuracy, output_difference = self.validate(
                    validation_set_inputs,
                    validation_set_outputs,
                    validation_data_indices
                )

                accuracy_by_epoch[0].append(epoch)
                accuracy_by_epoch[1].append(accuracy)

                if accuracy > 94.0:
                    _learning_rate = _second_learning_rate

                if accuracy >= minimum_accuracy and output_difference >= minimum_output_difference:
                    # If we met minimum accuracy, repeat the validation to make sure it wasn't a fluke.
                    for _ in range(5):
                        rep_accuracy, rep_output_difference = self.validate(
                            validation_set_inputs,
                            validation_set_outputs,
                            validation_data_indices
                        )
                    
                        accuracy = min(accuracy, rep_accuracy)

                        if accuracy < minimum_accuracy:
                            print('Repeated accuracy too low: ', accuracy)
                            break


                    if accuracy >= minimum_accuracy:
                        return accuracy_by_epoch, output_difference
            
            print(f'Epoch {epoch}: {accuracy:3.5}, minimum difference: {output_difference:2.5}')

        self.accuracy_by_epoch = accuracy_by_epoch
        
        return accuracy_by_epoch, output_difference


    def validate(self, test_inputs, test_outputs, indices):
        
        validate_by_software_model = False
        # validate_by_software_model = _validate_by_software_model

        correct_predictions = 0

        minimum_output_difference = float('inf')

        for index in indices:
            # if validate_by_software_model:
            #     outputs = self.software_think(array([test_inputs[index]]))
            # else:
            #     outputs = self.think(array([test_inputs[index]]), False)
            
            outputs = self.think(array([test_inputs[index]]), False)


            # print([outputs[-1]])
            # probabilities = softmax([outputs[-1]])
            prediction = np.argmax(outputs[-1],axis=1)[0]

            # print(f"{prediction}: {test_outputs[index]}")

            if test_outputs[index][prediction] == 1:
                correct_predictions += 1.0
                # pred_output = outputs[-1][0][prediction]
                # minimum_output_difference = min(minimum_output_difference, min([abs(x - pred_output) for i,x in enumerate(outputs[-1][0]) if i!=prediction]))

        # print(f'Training accuracy: {correct_predictions / len(indices) * 100.0}')

        return correct_predictions / len(indices) * 100.0, minimum_output_difference

    def think(self, inputs, update_weights=True):
        # use_software_model = _use_software_model
        # if use_software_model:
        #     return self.software_think(inputs)
        # else:
        #     hwt = self.hardware_think(inputs, update_weights)
        #     print('Hardware think():')
        #     pprint(hwt)

        #     swt = self.software_think(inputs)
        #     print('Software think():')
        #     pprint(swt)

        #     print('Error')
        #     pprint([hwt[i]-swt[i] for i in range(len(hwt))])
        #     print('\n')
        #     return hwt

        return self.hardware_think(inputs, update_weights)

    def hardware_think(self, inputs, update_weights=True):
        # print('Think() inputs:')
        # pprint(inputs)
        # print('\n')
        # print('Think() inputs.tolist():')
        # pprint(inputs.tolist())
        # print('\n')

        if update_weights:
            self.ckt.update_synaptic_weights()
        
        output_tensors = [[] for i in range(0, len(self.neuron_layers))]
        
        for input in inputs:
            network_output = self.ckt.think(input)

            for idx,output_tensor in enumerate(network_output):
                output_tensors[idx].append(output_tensor)

        return [np.asarray(x) for x in output_tensors]

    # The neural network thinks.
    def software_think(self, inputs, _ = True):
        layers = self.neuron_layers

        input_tensors = []
        output_tensors = []

        input_tensors.append(inputs)

        for i in range(0, len(layers)):
            output = self.activation_function(dot(input_tensors[i], layers[i].synaptic_weights))

            output_tensors.append(output)
            input_tensors.append(output)


        # print('Old think():')
        # pprint(output_tensors)

        return output_tensors

    # The neural network prints its weights
    def print_weights(self):
        layers = self.neuron_layers

        for i in range(0, len(layers)):
            layer = layers[i]
            print(f"    Layer {i} ({layer.neuron_count} neurons, each with {layer.inputs_per_neuron} inputs): ")
            print(layer.synaptic_weights)

def load_data(data_char_set):    
    data_chars = ''.join(data_char_set)

    try:
        with open(f'emnist_data_serialized{data_chars}.pickle', 'rb') as f:
            all_input_data = pickle.load(f)
    except:
        training_set_inputs, training_set_outputs, validation_set_inputs, validation_set_outputs = emnist_loader.load(_emnist_path, width, data_char_set)

        all_input_data = (training_set_inputs, training_set_outputs, validation_set_inputs, validation_set_outputs)

        with open(f'emnist_data_serialized{data_chars}.pickle', 'wb') as f:
            pickle.dump(all_input_data, f, pickle.HIGHEST_PROTOCOL)
    
    return all_input_data

if __name__ == "__main__":
    #Seed the random number generator
    random.seed(5)


    # Data image width (in pixels)
    width = 10

    # Characters used in data set
    data_char_set = _data_char_set

    # Neuron count for each hidden layer
    hidden_layer_sizes = [3]

    # Load data sets and create prediction labels
    
    all_input_data = load_data(data_char_set)
    training_set_inputs, training_set_outputs, validation_set_inputs, validation_set_outputs = all_input_data

    # Binarize data set.
    threshold = 30
    training_set_inputs = training_set_inputs > threshold
    validation_set_inputs = validation_set_inputs > threshold

    minimum_accuracy = _minimum_accuracy
    minimum_output_difference = _minimum_output_difference
    starting_seed = _starting_seed
    seed_iterations = _seed_iterations

    accuracy_by_epoch = [[1], [0.0]]

    for seed in range(starting_seed, starting_seed+seed_iterations):

        random.seed(seed)

        # Create neuron layers (M neurons, each with N inputs)
        #  (M for layer x must equal N for layer x+1)
        layer_dimensions = zip(hidden_layer_sizes + [len(data_char_set)], [width**2] + hidden_layer_sizes)
        neuron_layers = [ NeuronLayer(y, x) for y, x in layer_dimensions]
    
        # Combine the layers to create a neural network
        neural_network = NeuralNetwork(neuron_layers)

        print("Stage 1) Random starting synaptic weights: ")
        neural_network.print_weights()

        print("Stage 2) Train the network: ")
        print('')
        print(f'Dataset: {neural_network.data_char_set}')
        print('')
        print('')

        # Train the neural network for a specified number of epochs using the training set.
        accuracy_by_epoch, output_difference = neural_network.train(
            training_set_inputs,
            training_set_outputs,
            validation_set_inputs,
            validation_set_outputs,
            _args.epochs,
            minimum_accuracy,
            minimum_output_difference
        )

        print(f'{accuracy_by_epoch[0][-1]}: {accuracy_by_epoch[1][-1]}')

        if accuracy_by_epoch[1][-1] >= minimum_accuracy and output_difference >= minimum_output_difference:
            print(f'Last value of random number generation seed: {seed}')
            break

    print("Stage 3) Validation:")

    samples_per_column = 6
    sample_indices = []
    sample_inputs = []
    sample_labels = []
    sample_outputs = []
    sample_preds = []
    
    final_accuracy, final_min_difference = neural_network.validate(validation_set_inputs, validation_set_outputs, [x for x in range(len(validation_set_inputs))])
    neural_network.final_accuracy = final_accuracy

    print(f'Accuracy: {final_accuracy}%')
    print(f'Minimum difference: {final_min_difference}')
    print('')
    # n = 258
    # for i in range(samples_per_column):
    #     j = 0
    #     while j < min(len(data_char_set), len(validation_set_outputs)):
    #         if np.argmax(validation_set_outputs[n]) == j:
    #             sample_indices.append(n)
    #             sample_inputs.append(validation_set_inputs[n])
    #             sample_labels.append(data_char_set[j])
    #             sample_outputs.append(neural_network.think(sample_inputs[-1])[-1])
    #             sample_preds.append(data_char_set[np.argmax(sample_outputs[-1][-1])])
    #             j += 1
    #         n += 1
    
    # for idx in range(0, len(sample_inputs[0]), width):
    #     print(sample_inputs[0][idx:(idx+width)])

    # sample_size = 25
    # sample_inputs = validation_set_inputs[0:sample_size]
    # sample_labels = [data_char_set[x] for x in np.argmax(validation_set_outputs[0:sample_size],axis=1)]
    # sample_outputs = [neural_network.think(x)[-1] for x in sample_inputs]
    # sample_preds = [data_char_set[x] for x in np.argmax(sample_outputs, axis=1)]

    # pprint(sample_outputs)

    # for idx in range(0, len(sample_inputs[0]), width):
    #     print(sample_inputs[0][idx:(idx+width)])

    if _args.plot:
        plot_accuracy(accuracy_by_epoch)
        # plot_data_samples(sample_inputs, sample_labels, sample_preds, width, samples_per_column)
    
    # input_prompt = 'Get handwritten samples from photo? '
    # console_input = input(input_prompt)
    # while console_input[0].lower() == 'y':
    #     handwritten_chars = handwritten_samples.get_images()
    #     sample_size = len(handwritten_chars)
    #     sample_inputs = [np.array(x) for x in handwritten_chars]
    #     # [data_char_set[x] for x in np.argmax(validation_set_outputs[0:sample_size],axis=1)]
    #     sample_labels = None
    #     sample_outputs = [neural_network.think(x)[-1] for x in sample_inputs]
    #     sample_preds = [data_char_set[x] for x in np.argmax(sample_outputs, axis=1)]
    #     plot_data_samples(sample_inputs, sample_labels, sample_preds, width, 3)

    #     console_input = input(input_prompt)
    
    serialize_neural_network(neural_network, final_accuracy)

    try:
        neural_network.ckt.reset_network()
    except:
        print('Couldn\'t reset network.')
