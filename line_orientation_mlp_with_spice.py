# Adapted from code by Milo Spencer-Harper
# source: https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a

import argparse
# import emnist_loader
import ltspice_mlp
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from numpy import exp, array, random, dot, argmax
from pprint import pprint


parser = argparse.ArgumentParser(
    description='Train a multi-layer perceptron to detect the orientation of a line.',
    allow_abbrev=True)

parser.add_argument('epochs', type=int,
                    help='number of training epochs')
parser.add_argument('training_batch_size', type=int, nargs='?', default=10000,
                    help='specify a maximum data-set size for training batch')
parser.add_argument('-plot', action='store_true',
                    help='show a plot of the accuracy data by epoch')
parser.add_argument('-noisy_activation', action='store_true',
                    help='add simulated noise to the activation function')
parser.add_argument('-data_from_files', action='store_true',
                    help='load training and validation data from text files')
parser.add_argument('-software_model', action='store_true',
                    help='use a software-based neural network model (no SPICE simulation)')
parser.add_argument('-activation_test', action='store_true',
                    help='test a neuron\'s activation function')


_args = parser.parse_args()

_training_data_file = "training_set.txt"
_validation_data_file = "validation_set.txt"

_validate_by_software_model = False
_validation_iterations = 16
_validation_tick_interval = 2

_max_weight = 10.0
_learning_rate = 1.0

_emnist_path = os.path.join(os.getcwd(), 'emnist_data')

# Standard deviation for activation noise
_standard_deviation = 0.35

def plot_data_samples(X_train, y_labels, y_train, width):
    fig = plt.figure()
    for i in range(len(X_train)):
        two_d = (np.reshape(X_train[i], (width, width)) * 255).astype(np.uint8)
        plt.subplot(5,5,i+1)
        plt.tight_layout()
        plt.imshow(two_d, cmap='gray', interpolation='none')
        plt.title(f"{y_labels[i]} -> {y_train[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.show()

def plot_accuracy(accuracy_by_epoch):
    epoch, accuracy = accuracy_by_epoch
    plt.plot(epoch, accuracy)

    noise_label = f' (activation noise standard dev. {_standard_deviation})' if _args.noisy_activation else ''
    batch_size_label = f'training batch size {_args.training_batch_size}'
    plt.title(f'Prediction Accuracy by Training Epoch\n{noise_label}\n{batch_size_label}')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.show()

def plot_activation(inputs, outputs):
    plt.plot(inputs, outputs, '.')
    plt.title(f'Activation Function based on {"software" if _args.software_model else "hardware"} model')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.show()

def activation_test():
    input_count = 3
    neuron_layers = [NeuronLayer(1, input_count)]
    neural_network = NeuralNetwork(neuron_layers)
    neural_network.neuron_layers[0].synaptic_weights[0:input_count] = np.atleast_1d(neuron_layers[0].max_weight)

    input_start, input_end = (0, 2)
    sigmoid_start, sigmoid_end = (0, 2)
    non_sigmoid_tick_interval = 0.5
    sigmoid_tick_interval = 0.05
    points_per_sigmoid_tick = 1

    inputs = [np.linspace(input_start, input_end, int((input_end - input_start) / non_sigmoid_tick_interval)) for x in range(0, input_count)][0]

    for i in range (0, points_per_sigmoid_tick):
        inputs = np.append(inputs, np.linspace(sigmoid_start, sigmoid_end, int((sigmoid_end-sigmoid_start) / sigmoid_tick_interval)))

    input_tensors = [(x * np.ones((1, input_count)))[0] for x in inputs]

    outputs = neural_network.think(array(input_tensors))[0]

    plot_activation(inputs, [output[0] for output in outputs])

# Source: https://medium.com/@awjuliani/simple-softmax-in-python-tutorial-d6b4c4ed5c16
def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z))).T
    # return sm

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron, starting_weights=None):
        self.neuron_count = number_of_neurons
        self.inputs_per_neuron = number_of_inputs_per_neuron
        self.max_weight = _max_weight
        
        if starting_weights is None:            
            self.synaptic_weights = (2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1) * 1.0
        else:
            self.synaptic_weights = starting_weights

    def adjust_weights(self, adjustments):
        max_weight = self.max_weight

        self.synaptic_weights += adjustments * _learning_rate
        abs_weights = np.abs(self.synaptic_weights)
        if (abs_weights > max_weight).any():
            self.synaptic_weights *= _max_weight / (abs_weights).max()

class NeuralNetwork():
    def __init__(self, neuron_layers):
        self.neuron_layers = neuron_layers
        
        self.ckt = ltspice_mlp.MLP_Circuit(self)

    def activation_function(self, x):
        return self.sigmoid(x)

    def activation_derivative(self, x):
        return self.sigmoid_derivative(x)

    def relu(self, x):
        return x * (x > 0)

    def relu_derivative(self, x):
        return x > 0

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def sigmoid(self, x):
        if _args.noisy_activation == True:
            x += random.normal(0, _standard_deviation, x.shape)

        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, validation_set_inputs, validation_set_outputs, number_of_training_iterations):
        batch_size = _args.training_batch_size
        layers = self.neuron_layers

        accuracy_by_epoch = ([], [])
        
        validation_iterations = _validation_iterations
        validation_tick_interval = _validation_tick_interval
        if _args.data_from_files:
            validation_data_indices = [random.randint(0, len(validation_set_inputs)) for x in range(0, validation_iterations)]
        else:
            validation_data_indices = [x for x in range(0, len(validation_set_inputs))]

        last_layer_idx = len(layers) - 1

        for iteration in range(number_of_training_iterations):
            # Shuffle the training set
            data_set_size = training_set_inputs.shape[0]
            indices = np.random.permutation(data_set_size)[0:batch_size]
            # print(indices)
            training_set_inputs = training_set_inputs[indices]
            training_set_outputs = training_set_outputs[indices]

            # Pass the training set through the neural network.
            output_from_layers = self.think(training_set_inputs)

            deltas = {}
            errors = {}

            # We have to calculate the last layer's error value first.
            last_layer = layers[last_layer_idx]
            last_layer_output = output_from_layers[last_layer_idx]
            errors[last_layer] = training_set_outputs - last_layer_output
            deltas[last_layer] = errors[last_layer] * self.activation_derivative(last_layer_output)

            # Then we can loop through the rest of the layers in descending order
            #   and calculate their values.
            for i in range(last_layer_idx - 1, -1, -1):
                layer = layers[i]
                output = output_from_layers[i]

                next_layer = layers[i+1]
                # next_layer_error = errors[next_layer]
                # next_layer_delta = deltas[next_layer]

                errors[layer] = deltas[next_layer].dot(next_layer.synaptic_weights.T)
                deltas[layer] = errors[layer] * self.activation_derivative(output)

            # Calculate how much to adjust the weights by
            adjustments = {}

            adjustments[layers[0]] = training_set_inputs.T.dot(deltas[layers[0]])

            for i in range (1, len(layers)):
                adjustments[layers[i]] = output_from_layers[i-1].T.dot(deltas[layers[i]])

            # Adjust the weights.
            for layer in layers:
                layer.adjust_weights(adjustments[layer])

            # Validate results
            if iteration % validation_tick_interval == 0:
                accuracy = self.validate(
                    validation_set_inputs,
                    validation_set_outputs,
                    validation_data_indices
                )

                accuracy_by_epoch[0].append(iteration)
                accuracy_by_epoch[1].append(accuracy)

            print(f'Epoch {iteration}: {accuracy}')

        return accuracy_by_epoch

    def validate(self, test_inputs, test_outputs, indices):
        
        validate_by_software_model = _validate_by_software_model

        correct_predictions = 0

        for index in indices:
            if validate_by_software_model:
                outputs = self.software_think(array([test_inputs[index]]))
            else:
                outputs = self.think(array([test_inputs[index]]))

            # print([outputs[-1]])
            # probabilities = softmax([outputs[-1]])
            prediction = np.argmax(outputs[-1],axis=1)[0]
            # print(f"{prediction}: {test_outputs[index]}")

            if test_outputs[index][prediction] == 1:
                correct_predictions += 1.0

        # print(f'Training accuracy: {correct_predictions / len(indices) * 100.0}')

        return correct_predictions / len(indices) * 100.0

    def think(self, inputs):
        use_software_model = _args.software_model
        if use_software_model:
            return self.software_think(inputs)
        else:
            return self.hardware_think(inputs)

    # The neural network thinks.
    def hardware_think(self, inputs):
        # print('Think() inputs:')
        # pprint(inputs)
        # print('\n')
        # print('Think() inputs.tolist():')
        # pprint(inputs.tolist())
        # print('\n')
        
        output_tensors = [[] for i in range(0, len(self.neuron_layers))]
        
        for input in inputs:
            network_output = self.ckt.think(input)

            for idx,output_tensor in enumerate(network_output):
                output_tensors[idx].append(output_tensor)

        # print('New think():')
        # pprint([np.asarray(x) for x in output_tensors])

        # self.old_think(inputs)
        return [np.asarray(x) for x in output_tensors]

    #DEBUG
    def software_think(self, inputs):
        layers = self.neuron_layers

        input_tensors = []
        output_tensors = []

        input_tensors.append(inputs)

        for i in range(0, len(layers)):
            output = self.activation_function(dot(input_tensors[i] * 2 - 0.2, layers[i].synaptic_weights))

            output_tensors.append(output)
            input_tensors.append(output)
        
        # print('Old think():')
        # pprint(output_tensors)

        return output_tensors

    # The neural network prints its weights
    def print_weights(self):
        layers = self.neuron_layers
        lines = []

        for i in range(0, len(layers)):
            layer = layers[i]
            lines.append(f'Layer {i} ({layer.neuron_count} neurons, each with {layer.inputs_per_neuron} inputs):\n')
            lines.append(f'{layer.synaptic_weights}\n')
        
        for x in lines:
            print(x)
        
        with open('starting_weights.txt', 'w') as f:
            f.writelines(lines)

def load_data(filename):
    output_dict = {
        "V" : [1, 0, 0],
        "H" : [0, 1, 0],
        "D" : [0, 0, 1]
    }

    inputs = []
    outputs = []

    with open(filename, 'r') as f:
        data_lines = f.readlines()

        for line in data_lines:
            data = line.split()

            outputs.append(output_dict[data[0]])
            inputs.append([float(x) for x in data[1:]])

    #DEBUG
    # print(inputs[0])
    # print(outputs[0])

    return array(inputs), array(outputs)

if __name__ == "__main__":

    #Seed the random number generator
    # random.seed(2)

    if _args.activation_test:
        activation_test()
        sys.exit(0)

    if _args.data_from_files:
        # Load training and validation data from files
        training_set_inputs, training_set_outputs = load_data(_training_data_file)
        validation_set_inputs, validation_set_outputs = load_data(_validation_data_file)
    else:
        # The standard training set. We have 7 examples, each consisting of 3 input values
        # and 1 output value.
        training_set_inputs = array(
            [
                [1, 0, 1, 0], #V
                [0, 1, 0, 1], #V
                [1, 1, 0, 0], #H
                [0, 0, 1, 1], #H
                [1, 0, 0, 1], #D
                [0, 1, 1, 0]  #D
            ]
        )

        training_set_outputs = array(
            [
                [1, 0, 0], #V
                [1, 0, 0], #V
                [0, 1, 0], #H
                [0, 1, 0], #H
                [0, 0, 1], #D
                [0, 0, 1]  #D
            ]
        )

        # In this case, training data and validation data are the same
        validation_set_inputs, validation_set_outputs = training_set_inputs, training_set_outputs

    prediction_labels = {
        0 : "Vertical",
        1 : "Horizontal",
        2 : "Diagonal"
    }

    starting_weights = [
        np.asarray(
            [[ 0.78693466,  0.00260829, -0.52798853,  0.92084359],
             [ 0.85578172,  0.50035073,  0.14404296, -0.03027512],
             [-0.43132144, -0.26774606, -0.89238543, -0.86610821],
             [ 0.87852929,  0.08643093, -0.51677491, -0.03856048]]),
        np.asarray(            
            [[-0.36415303,  0.40740717,  0.50542251, -0.59614972],
             [ 0.59996569, -0.69599694, -0.74805606,  0.61728582],
             [ 0.50074242, -0.85998174, -0.33990181, -0.50978041],
             [-0.92183461,  0.90926905, -0.44542875, -0.09638122]]),
        np.asarray(
            [[-0.11044091, -0.67848051, -0.70275169],
             [-0.65093601,  0.4998441,   0.95182468],
             [-0.62298891, -0.00173666,  0.59165978],
             [-0.72568407,  0.90145892,  0.39221916]])
    ]

    # Create neuron layers (M neurons, each with N inputs)
    #  (M for layer x must equal N for layer x+1)
    neuron_layers = [
        NeuronLayer(4, 4, starting_weights[0]),
        NeuronLayer(4, 4, starting_weights[1]),
        NeuronLayer(3, 4, starting_weights[2])]

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(neuron_layers)

    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # Train the neural network for a specified number of epochs using the training set.
    accuracy_by_epoch = neural_network.train(
        training_set_inputs,
        training_set_outputs,
        validation_set_inputs,
        validation_set_outputs,
         _args.epochs
    )

    # print("Stage 2) New synaptic weights after training: ")
    # neural_network.print_weights()

    if _args.plot:
        plot_accuracy(accuracy_by_epoch)

    print("Stage 3) Validation:")
    for input_set in validation_set_inputs[0:6]:
        ticks = ["X" if x > 0.5 else " " for x in input_set]

        outputs = neural_network.think(array([input_set]))
        # probs = softmax([outputs[-1]])
        preds = np.argmax([outputs[-1][0]],axis=1)

        print(f"           _______________       ")
        print(f"          |       |       |      Prediction: {prediction_labels[preds[0]]}")
        print("          |   {}   |   {}   |".format(ticks[0], ticks[1]))
        print("          |_______|_______|")
        print("          |       |       |")
        print("          |   {}   |   {}   |".format(ticks[2], ticks[3]))
        print("          |_______|_______|\n\n")
