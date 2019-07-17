# Adapted from code by Milo Spencer-Harper
# source: https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a

import numpy as np
from numpy import exp, array, random, dot, argmax

# Source: https://medium.com/@awjuliani/simple-softmax-in-python-tutorial-d6b4c4ed5c16
def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layer0, layer1, layer2):
        self.layer0 = layer0
        self.layer1 = layer1
        self.layer2 = layer2

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_0, output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate the error for layer 0 (By looking at the weights in layer 0,
            # we can determine by how much layer 0 contributed to the error in layer 1).
            layer0_error = layer1_delta.dot(self.layer1.synaptic_weights.T)
            layer0_delta = layer0_error * self.__sigmoid_derivative(output_from_layer_0)

            # Calculate how much to adjust the weights by
            layer0_adjustment = training_set_inputs.T.dot(layer0_delta)
            layer1_adjustment = output_from_layer_0.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer0.synaptic_weights += layer0_adjustment
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer0 = self.__sigmoid(dot(inputs, self.layer0.synaptic_weights))
        output_from_layer1 = self.__sigmoid(dot(output_from_layer0, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer0, output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print("    Layer 0 (4 neurons, each with 4 inputs): ")
        print(self.layer0.synaptic_weights)
        print("    Layer 1 (4 neurons, each with 4 inputs): ")
        print(self.layer1.synaptic_weights)
        print("    Layer 2 (3 neurons, with 4 inputs):")
        print(self.layer2.synaptic_weights)

if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)

    # Create neuron layers (M neurons, each with N inputs)
    layer0 = NeuronLayer(4, 4)
    layer1 = NeuronLayer(4, 4)
    layer2 = NeuronLayer(3, 4)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer0, layer1, layer2)

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
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.

    print("Stage 3) Validation:")
    for input_set in training_set_inputs:
        ticks = ["X" if x == 1 else " " for x in input_set]

        hidden_state0, hidden_state1, output = neural_network.think(array(input_set))
        probs = softmax([output])
        preds = np.argmax(probs,axis=1)

        print(f"           _______________       Probabilities: {probs}")
        print(f"          |       |       |      Prediction: {prediction_labels[preds[0]]}")
        print("          |   {}   |   {}   |".format(ticks[0], ticks[1]))
        print("          |_______|_______|")
        print("          |       |       |")
        print("          |   {}   |   {}   |".format(ticks[2], ticks[3]))
        print("          |_______|_______|\n\n")
