# Adapted from code by Milo Spencer-Harper
# source: https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a

import argparse
import emnist_loader
import ltspice_mlp
import matplotlib.pyplot as plt
import numpy as np
import os
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
# parser.add_argument('-data_from_files', action='store_true',
#                     help='load training and validation data from text files')

_args = parser.parse_args()

# _training_data_file = "training_set.txt"
# _validation_data_file = "validation_set.txt"

_validation_iterations = 200
_validation_tick_interval = 1
_max_weight = 10.0

_emnist_path = os.path.join(os.getcwd(), 'emnist_data')

# Standard deviation for activation noise
_standard_deviation = 0.45

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

# Source: https://medium.com/@awjuliani/simple-softmax-in-python-tutorial-d6b4c4ed5c16
def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z))).T
    # return sm

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.neuron_count = number_of_neurons
        self.inputs_per_neuron = number_of_inputs_per_neuron
        self.max_weight = _max_weight
        
        self.synaptic_weights = (2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1) / self.max_weight

    def adjust_weights(self, adjustments):
        max_weight = self.max_weight

        self.synaptic_weights += adjustments
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
        return 0.5 * x * (x > 0)

    def relu_derivative(self, x):
        return 0.5 * (x > 30)

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
        validation_data_indices = [random.randint(0, len(validation_set_inputs)) for x in range(0, validation_iterations)]

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
                next_layer_error = errors[next_layer]
                next_layer_delta = deltas[next_layer]

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
        correct_predictions = 0

        for index in indices:
            # print('\nValidation test_inputs[index]')
            # pprint(test_inputs[index])
            # print('\n')
            # print('\nValidation array(test_inputs[index])')
            # pprint(array(test_inputs[index]))
            # print('\n')
            # print('\nValidation array(test_inputs[index]).tolist()')
            # pprint(array(test_inputs[index]).tolist())
            # print('\n')
            outputs = self.think(array([test_inputs[index]]))

            # print([outputs[-1]])
            # probabilities = softmax([outputs[-1]])
            prediction = np.argmax(outputs[-1],axis=1)[0]
            # print(f"{prediction}: {test_outputs[index]}")

            if test_outputs[index][prediction] == 1:
                correct_predictions += 1.0

        # print(f'Training accuracy: {correct_predictions / len(indices) * 100.0}')

        return correct_predictions / len(indices) * 100.0


    # The neural network thinks.
    def think(self, inputs):
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

        self.old_think(inputs)
        return [np.asarray(x) for x in output_tensors]

    #DEBUG
    def old_think(self, inputs):
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

if __name__ == "__main__":
    # Data image width (in pixels)
    width = 5

    #Seed the random number generator
    random.seed(1)

    training_set_inputs, training_set_outputs, validation_set_inputs, validation_set_outputs = emnist_loader.load(_emnist_path, width)

    prediction_labels = {
        0 : "X",
        1 : "O"
    }

    # Create neuron layers (M neurons, each with N inputs)
    #  (M for layer x must equal N for layer x+1)
    neuron_layers = [
        NeuronLayer(2, width*width)
    ]

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(neuron_layers)

    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # Binarize data set.
    training_set_inputs = neural_network.relu_derivative(training_set_inputs)
    validation_set_inputs = neural_network.relu_derivative(validation_set_inputs)

    # Train the neural network for a specified number of epochs using the training set.
    accuracy_by_epoch = neural_network.train(
        training_set_inputs,
        training_set_outputs,
        validation_set_inputs,
        validation_set_outputs,
         _args.epochs
    )

    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    if _args.plot:
        plot_accuracy(accuracy_by_epoch)

    print("Stage 3) Validation:")
    sample_size = 25

    sample_inputs = validation_set_inputs[0:sample_size]

    sample_labels = [prediction_labels[x] for x in np.argmax(validation_set_outputs[0:sample_size],axis=1)]

    sample_outputs = [neural_network.think([x])[-1][0].tolist() for x in sample_inputs]

    # print([x for x in sample_inputs])
    # print(sample_outputs)
    # print([prediction_labels[x] for x in np.argmax(sample_outputs, axis=1)])

    print('Sample inputs:')
    pprint(sample_inputs)
    print('Sample outputs:')
    pprint(sample_outputs)

    print('np.argmax(sample_outputs, axis=1):')
    pprint(np.argmax(sample_outputs, axis=1))

    sample_preds = [prediction_labels[x] for x in np.argmax(sample_outputs, axis=1)]

    print('Sample preds:')
    pprint(sample_preds)

    if _args.plot:
        plot_data_samples(sample_inputs, sample_labels, sample_preds, width)
