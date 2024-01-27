from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def define_dense_model_with_hidden_layers(input_length, 
                                          activation_func_array=['sigmoid','sigmoid'],
                                          hidden_layers_sizes=[50, 20],
                                          output_function='softmax',
                                          output_length=10):
    """Define a dense model with multiple hidden layers.
    input_length: the number of inputs for the first layer
    activation_func_array: an array of activation functions for the hidden layers
    hidden_layers_sizes: an array of the number of neurons in each hidden layer
    output_function: the activation function for the output layer
    output_length: the number of outputs (number of neurons in the output layer)"""

    model = keras.Sequential()

    # Create the input layer
    model.add(layers.Dense(hidden_layers_sizes[0], activation=activation_func_array[0], input_shape=(input_length,)))
    
    # Create the hidden layers
    for i in range(1, len(hidden_layers_sizes)):
        model.add(layers.Dense(hidden_layers_sizes[i], activation=activation_func_array[i]))
    
    # Create the output layer
    model.add(layers.Dense(output_length, activation=output_function))
    return model

def set_layers_to_trainable(model, trainable_layer_numbers):
    """Set specific layers of the model as trainable or non-trainable.
    model: the model
    trainable_layer_numbers: a list of layer numbers to be set as trainable. 
    Other layers will be set as non-trainable."""
    for i, layer in enumerate(model.layers):
        layer.trainable = i in trainable_layer_numbers
    return model