from line_orientation_mlp_with_i2c import NeuronLayer, NeuralNetwork

import matplotlib.pyplot as plt
import numpy as np

def plot_activation(input_list, output_lists, neuron_names):
    for output in output_lists:
        plt.plot(input_list, output)

    plt.title('Activation function output vs. weight (all inputs at max value)')
    plt.xlabel('Byte value of weight')
    plt.ylabel('ADC output reading')
    plt.legend(neuron_names)

    plt.show()

    
def main():
    layers = [NeuronLayer(4,4), NeuronLayer(3,4)]

    for layer in layers:
        for weights in layer.synaptic_weights:
            for i in range(len(weights)):
                weights[i] = -layer.max_weight

    MLP = NeuralNetwork(layers)

    max_weight = layers[0].max_weight 

    neuron_names = []
    for i in range(len(layers)):
        for j in range(layers[i].neuron_count):
            neuron_names.append(f'Neuron {i}_{j}')
    
    neuron_names.append('Software model')

    output_lists = [[] for i in range(1 + sum([x.neuron_count for x in layers]))]

    inputs = []
    
    tick_count = 100
    tick_size = max_weight*2 / tick_count

    start_idx = 0
    for i,layer in enumerate(layers):
        end_idx = start_idx + layer.neuron_count + (1 if i==(len(layers)-1) else 0)
        
        for synapse in layer.synaptic_weights:
            for ticks in range(tick_count):
                #weights[i][0:][synapse] = weight
                for idx in range(len(synapse)):
                    synapse[idx] = -layer.max_weight + ticks*tick_size
                print(layer.synaptic_weights)
            
                outputs = MLP.think([np.array(np.array([1,0,0,0]))])
                ideal_outputs = MLP.software_think(np.array(np.array([128,128,128,128])))
            
                idx = start_idx
                for j in range(layer.neuron_count):
                    output_lists[idx].append(outputs[i][0][j])
                    idx += 1
                if (i == (len(layers)-1)):
                    output_lists[idx].append(ideal_outputs[i][0])
                    inputs.append(sum([synapse[0] for synapse in layer.synaptic_weights]))
                    
        start_idx = end_idx
   
   # plot_activation(inputs, output_lists, neuron_names)
    plot_activation([x for x in range(tick_count*4)], output_lists, neuron_names)

if __name__ == '__main__':
    main()
