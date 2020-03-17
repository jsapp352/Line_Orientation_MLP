from emnist_mlp import *
import handwritten_samples
import numpy as np

# Load saved MLP object
neural_network = deserialize_neural_network('emnist_mlp_UCF_2020_03_14_16_03_12.pickle')
data_char_set = neural_network.data_char_set
width = 10

# Get character input
input_prompt = 'Get handwritten samples from photo? '
console_input = input(input_prompt)
while console_input[0].lower() == 'y':
        handwritten_chars = handwritten_samples.get_images()
        sample_size = len(handwritten_chars)
        sample_inputs = [np.array(x) for x in handwritten_chars]
        # [data_char_set[x] for x in np.argmax(validation_set_outputs[0:sample_size],axis=1)]
        sample_labels = None
        sample_outputs = [neural_network.think(x)[-1] for x in sample_inputs]
        sample_preds = [data_char_set[x]
                        for x in np.argmax(sample_outputs, axis=1)]
        plot_data_samples(sample_inputs, sample_labels, sample_preds, width, 3)

        console_input = input(input_prompt)

# Get network prediction and response time


# Display prediction and response time

