# Adapted from code by Milo Spencer-Harper
# source: https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a

import argparse
import numpy as np
from numpy import exp, array, random, dot, argmax

parser = argparse.ArgumentParser(description='Train a multi-layer perceptron to detect the orientation of a line.')
parser.add_argument('epochs', type=int,
                    help='number of training epochs')

_args = parser.parse_args()

# Source: https://medium.com/@awjuliani/simple-softmax-in-python-tutorial-d6b4c4ed5c16
def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.neuron_count = number_of_neurons
        self.inputs_per_neuron = number_of_inputs_per_neuron
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, neuron_layers):
        self.neuron_layers = neuron_layers

    def activation_function(self, x):
        return self.sigmoid(x)

    def activation_derivative(self, x):
        return self.sigmoid_derivative(x)

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        layers = self.neuron_layers

        for iteration in range(number_of_training_iterations):
            # Pass the training set through the neural network.
            output_from_layers = self.think(training_set_inputs)

            last_layer_idx = len(neuron_layers) - 1
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
                layer.synaptic_weights += adjustments[layer]

    # The neural network thinks.
    def think(self, inputs):
        layers = self.neuron_layers

        input_tensors = []
        output_tensors = []

        input_tensors.append(inputs)

        for i in range(0, len(layers)):
            output = self.activation_function(dot(input_tensors[i], layers[i].synaptic_weights))

            output_tensors.append(output)
            input_tensors.append(output)

        return output_tensors

    # The neural network prints its weights
    def print_weights(self):
        layers = self.neuron_layers

        for i in range(0, len(layers)):
            layer = layers[i]
            print(f"    Layer {i} ({layer.neuron_count} neurons, each with {layer.inputs_per_neuron} inputs): ")
            print(layer.synaptic_weights)

if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)

    # Create neuron layers (M neurons, each with N inputs)
    neuron_layers = [
        NeuronLayer(2, 4),
        NeuronLayer(4, 2),
        NeuronLayer(3, 4)]

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(neuron_layers)

    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
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

    prediction_labels = {
        0 : "Vertical",
        1 : "Horizontal",
        2 : "Diagonal"
    }

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, _args.epochs)

    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.

    print("Stage 3) Validation:")
    for input_set in training_set_inputs:
        ticks = ["X" if x == 1 else " " for x in input_set]

        outputs = neural_network.think(array(input_set))
        probs = softmax([outputs[-1]])
        preds = np.argmax(probs,axis=1)

        print(f"           _______________       Probabilities: {probs}")
        print(f"          |       |       |      Prediction: {prediction_labels[preds[0]]}")
        print("          |   {}   |   {}   |".format(ticks[0], ticks[1]))
        print("          |_______|_______|")
        print("          |       |       |")
        print("          |   {}   |   {}   |".format(ticks[2], ticks[3]))
        print("          |_______|_______|\n\n")
