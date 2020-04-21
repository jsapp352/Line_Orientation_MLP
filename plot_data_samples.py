import emnist_mlp
from emnist_mlp import NeuralNetwork, NeuronLayer
import numpy as np
from numpy import array
from time import sleep
import sys
import timeit
import matplotlib.pyplot as plt

filename = 'emnist_mlp_CAT_2020_04_12_20_49_43_97p9.pickle'

if len(sys.argv) > 1:
    filename = sys.argv[1]

neural_network = emnist_mlp.deserialize_neural_network(filename)

all_input_data = emnist_mlp.load_data(neural_network.data_char_set)

training_set_inputs, training_set_outputs, validation_set_inputs, validation_set_outputs = all_input_data

# Binarize data set.
threshold = 30
training_set_inputs = training_set_inputs > threshold
validation_set_inputs = validation_set_inputs > threshold

data_char_set = neural_network.data_char_set

samples_per_column = 7
sample_indices = []
sample_inputs = []
sample_labels = []
sample_outputs = []
sample_preds = []

n = 28
for i in range(samples_per_column):
    j = 0
    while j < min(len(data_char_set), len(validation_set_outputs)):
        if np.argmax(validation_set_outputs[n]) == j:
            sample_indices.append(n)
            sample_inputs.append(validation_set_inputs[n])
            sample_labels.append(data_char_set[j])

            prediction_result = neural_network.think([sample_inputs[-1]])[-1][0]

            prediction_idx = np.argmax(prediction_result)

            sample_preds.append(data_char_set[prediction_idx])
            j += 1
        n += 1

print(sample_preds)
width = 10
emnist_mlp.plot_data_samples(sample_inputs, sample_labels, sample_preds, width, samples_per_column)