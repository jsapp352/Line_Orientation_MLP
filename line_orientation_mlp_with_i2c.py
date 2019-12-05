# Adapted from code by Milo Spencer-Harper
# source: https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a

import argparse
# import emnist_loader
from hardwareMLP import MLP_Circuit
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from numpy import exp, array, random, dot, argmax
from pprint import pprint, pformat


parser = argparse.ArgumentParser(
    description='Train a multi-layer perceptron to detect the orientation of a line.',
    allow_abbrev=True)

parser.add_argument('epochs', type=int, nargs='?', default=1,
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

_max_weight = 1.0
# _learning_rate = 0.04
_learning_rate = 0.01

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
    input_count = 4
    neuron_layers = [NeuronLayer(4,4), NeuronLayer(3,4)]
    neural_network = NeuralNetwork(neuron_layers)

    for layer in neural_network.neuron_layers:
        for synapse in layer.synaptic_weights:
            for weight in synapse:
                print(f'{weight} -> ')
                weight = layer.max_weight
                print(f'{weight}\n')

    #neural_network.neuron_layers[0].synaptic_weights[0:input_count] = list(np.atleast_1d(neuron_layers[0].max_weight))

    input_start, input_end = (-5, 5)
    sigmoid_start, sigmoid_end = (-1, 1)
    non_sigmoid_tick_interval = 0.5
    sigmoid_tick_interval = 0.05
    points_per_sigmoid_tick = 1

    inputs = [np.linspace(input_start, input_end, int((input_end - input_start) / non_sigmoid_tick_interval)) for x in range(0, input_count)][0]

    for i in range (0, points_per_sigmoid_tick):
        inputs = np.append(inputs, np.linspace(sigmoid_start, sigmoid_end, int((sigmoid_end-sigmoid_start) / sigmoid_tick_interval)))

    input_tensors = [(x * np.ones((1, input_count)))[0] for x in inputs]

    # print(input_tensors[0])

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
            # self.synaptic_weights = (2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1) * 0.5
            self.synaptic_weights = random.normal(size=(number_of_inputs_per_neuron, number_of_neurons), scale=(self.inputs_per_neuron**-0.5))
        else:
            self.synaptic_weights = starting_weights

    def adjust_weights(self, adjustments):
        max_weight = self.max_weight

        self.synaptic_weights += adjustments * _learning_rate
        abs_weights = np.abs(self.synaptic_weights)
        if (abs_weights > max_weight).any():
            self.synaptic_weights *= max_weight / (abs_weights).max()

class NeuralNetwork():
    def __init__(self, neuron_layers):
        self.neuron_layers = neuron_layers

        # self.activation_function = self.sigmoid
        # self.activation_derivative = self.sigmoid_derivative

        self.activation_function = self.tanh
        self.activation_derivative = self.tanh_derivative
        
        self.ckt = MLP_Circuit(self)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1.0 - np.tanh(x)**2

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
    def train(self, training_set_inputs, training_set_outputs, validation_set_inputs, validation_set_outputs, number_of_training_iterations, minimum_accuracy):
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
            indices = np.asarray([x for x in range(data_set_size)])# np.random.permutation(data_set_size)[0:batch_size]
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

                if accuracy >= minimum_accuracy:
                    return accuracy_by_epoch

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
           # hwt = self.hardware_think(inputs)
           # print('Hardware think():')
           # pprint(hwt)

            #swt = self.software_think(inputs)
            #print('Software think():')
            #pprint(swt)

            #print('Error')
            #pprint([hwt[i]-swt[i] for i in range(len(hwt))])
            #return hwt

            return self.hardware_think(inputs)

    # The neural network thinks.
    def hardware_think(self, inputs):
        # print('Think() inputs:')
        # pprint(inputs)
        # print('\n')
        # print('Think() inputs.tolist():')
        # pprint(inputs.tolist())
        # print('\n')

        self.ckt.update_synaptic_weights()
        
        output_tensors = [[] for i in range(0, len(self.neuron_layers))]
        
        for input in inputs:
            network_output = self.ckt.think(input)

            for idx,output_tensor in enumerate(network_output):
                output_tensors[idx].append(output_tensor)

        return [np.asarray(x) for x in output_tensors]

    #DEBUG
    def software_think(self, inputs):
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
    def print_weights(self, filename=None):
        layers = self.neuron_layers
        lines = []

        for i in range(0, len(layers)):
            layer = layers[i]
            lines.append(f'Layer {i} ({layer.neuron_count} neurons, each with {layer.inputs_per_neuron} inputs):\n')
            lines.append(f'{pformat(layer.synaptic_weights.tolist(), width=100)}\n')
        
        for x in lines:
            print(x)
        
        if filename != None:
            with open(filename, 'w') as f:
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
    random.seed(5) # HW setting
    # random.seed(8) # SW setting

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
        training_set_inputs_sig = array(
            [
                [0, 1, 0, 1],  # V
                [1, 0, 1, 0],  # V
                [0, 0, 1, 1],  # H
                [1, 1, 0, 0],  # H
                [0, 1, 1, 0],  # D
                [1, 0, 0, 1],  # D
            ]
        )

        training_set_inputs_tanh = array(
            [
                [-1,  1, -1,  1],  # V
                [ 1, -1,  1, -1],  # V
                [-1, -1,  1,  1],  # H
                [ 1,  1, -1, -1],  # H
                [-1,  1,  1, -1],  # D
                [ 1, -1, -1,  1],  # D
            ]
        )

        training_set_outputs_sig = array(
            [
                [1, 0, 0],  # V
                [1, 0, 0],  # V
                [0, 1, 0],  # H
                [0, 1, 0],  # H
                [0, 0, 1],  # D
                [0, 0, 1]   # D
            ]
        )

        training_set_outputs_tanh = array(
            [
                [1, -1, -1,],  # V
                [1, -1, -1,],  # V
                [-1,  1, -1],  # H
                [-1,  1, -1],  # H
                [-1, -1,  1],  # D
                [-1, -1,  1]   # D
            ]
        )

        training_set_inputs_sig4 = array(
            [
                [0, 1, 0, 1],  # V
                [1, 0, 1, 0],  # V
                [0, 0, 1, 1],  # H
                [1, 1, 0, 0],  # H
                [0, 1, 1, 0],  # D
                [1, 0, 0, 1],  # D
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 0],
                [1, 1, 1, 1]
            ]
        )

        training_set_inputs_tanh4 = array(
            [
                [-1,  1, -1,  1],  # V
                [ 1, -1,  1, -1],  # V
                [-1, -1,  1,  1],  # H
                [ 1,  1, -1, -1],  # H
                [-1,  1,  1, -1],  # D
                [ 1, -1, -1,  1],  # D
                [-1, -1, -1, -1],
                [-1, -1, -1,  1],
                [-1, -1,  1, -1],
                [-1,  1, -1, -1],
                [-1,  1,  1,  1],
                [ 1, -1, -1, -1],
                [ 1, -1,  1,  1],
                [ 1,  1, -1,  1],
                [ 1,  1,  1, -1],
                [ 1,  1,  1,  1]
            ]
        )

        training_set_outputs_sig4 = array(
        [
            [1, 0, 0, 0],  # V
            [1, 0, 0, 0],  # V
            [0, 1, 0, 0],  # H
            [0, 1, 0, 0],  # H
            [0, 0, 1, 0],  # D
            [0, 0, 1, 0],  # D
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ]
        )

        training_set_outputs_tanh4 = array(
            [
                [ 1, -1, -1, -1],  # V
                [ 1, -1, -1, -1],  # V
                [-1,  1, -1, -1],  # H
                [-1,  1, -1, -1],  # H
                [-1, -1,  1, -1],  # D
                [-1, -1,  1, -1],  # D
                [-1, -1, -1,  1],
                [-1, -1, -1,  1],
                [-1, -1, -1,  1],
                [-1, -1, -1,  1],
                [-1, -1, -1,  1],
                [-1, -1, -1,  1],
                [-1, -1, -1,  1],
                [-1, -1, -1,  1],
                [-1, -1, -1,  1],
                [-1, -1, -1,  1]
            ]
        )

        training_set_inputs = training_set_inputs_sig
        training_set_outputs = training_set_outputs_sig

        # training_set_inputs = training_set_inputs_tanh
        training_set_outputs = training_set_outputs_tanh

        # In this case, training data and validation data are the same
        validation_set_inputs, validation_set_outputs = training_set_inputs, training_set_outputs

    prediction_labels = {
        0 : "Vertical",
        1 : "Horizontal",
        2 : "Diagonal",
        3 : "None"
    }

    # starting_weights = [
    #     np.asarray(
    #         [[ 0.78693466,  0.00260829, -0.52798853,  0.92084359],
    #          [ 0.85578172,  0.50035073,  0.14404296, -0.03027512],
    #          [-0.43132144, -0.26774606, -0.89238543, -0.86610821],
    #          [ 0.87852929,  0.08643093, -0.51677491, -0.03856048]]),
    #     np.asarray(            
    #         [[-0.36415303,  0.40740717,  0.50542251, -0.59614972],
    #          [ 0.59996569, -0.69599694, -0.74805606,  0.61728582],
    #          [ 0.50074242, -0.85998174, -0.33990181, -0.50978041],
    #          [-0.92183461,  0.90926905, -0.44542875, -0.09638122]]),
    #     np.asarray(
    #         [[-0.11044091, -0.67848051, -0.70275169],
    #          [-0.65093601,  0.4998441,   0.95182468],
    #          [-0.62298891, -0.00173666,  0.59165978],
    #          [-0.72568407,  0.90145892,  0.39221916]])
    # ]

    starting_weights = [
        np.asarray(
            [[-0.46630699, -0.29129736,  0.50830302,  0.89999705],
             [ 0.00281956, -0.79126006,  0.04567647, -0.07229544],
             [ 0.53333171, -0.44043369,  0.10299210, -0.88182197],
             [-0.10828991, -0.91825339,  0.60303642, -0.47984214]]
        ),
        np.asarray(
            [[-0.93293758, -0.57259914,  0.85332560],
             [ 0.884597470, 0.17658923, -0.53673408],
             [-0.38296133, -0.79041703,  0.90380022],
             [ 0.910205120,-0.93333956,  0.71895160]]
        )
    ]

    starting_weights = [None, None]

    # Neuron count for each hidden layer
    hidden_layer_sizes = [4]

    minimum_accuracy = 199.0

    accuracy_by_epoch = [[1], [0.0]]
    while accuracy_by_epoch[1][-1] < minimum_accuracy:

        # Create neuron layers (M neurons, each with N inputs)
        #  (M for layer x must equal N for layer x+1)
        layer_dimensions = zip(
            hidden_layer_sizes + [len(training_set_outputs[0])], 
            [len(training_set_inputs[0])] + hidden_layer_sizes,
            starting_weights)


        neuron_layers = [NeuronLayer(y, x) for y, x, w in layer_dimensions]

        # Combine the layers to create a neural network
        neural_network = NeuralNetwork(neuron_layers)

        print("Stage 1) Random starting synaptic weights: ")
        neural_network.print_weights('starting_weights.txt')

        # Train the neural network for a specified number of epochs using the training set.
        accuracy_by_epoch = neural_network.train(
            training_set_inputs,
            training_set_outputs,
            validation_set_inputs,
            validation_set_outputs,
            _args.epochs,
            minimum_accuracy
        )

        print(f'{accuracy_by_epoch[0][-1]}: {accuracy_by_epoch[1][-1]}')

        break

    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    if _args.plot:
        plot_accuracy(accuracy_by_epoch)

    print("Stage 3) Validation:")
    for input_set in validation_set_inputs[0:6]:
        ticks = ["X" if x > 0.5 else " " for x in input_set]

        outputs = neural_network.think(array([input_set]))

        print(outputs[-1][0])
        # probs = softmax([outputs[-1]])
        preds = np.argmax([outputs[-1][0]],axis=1)

        print(f"           _______________       ")
        print(f"          |       |       |      Prediction: {prediction_labels[preds[0]]}")
        print("          |   {}   |   {}   |".format(ticks[0], ticks[1]))
        print("          |_______|_______|")
        print("          |       |       |")
        print("          |   {}   |   {}   |".format(ticks[2], ticks[3]))
        print("          |_______|_______|\n\n")
