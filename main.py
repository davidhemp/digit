#!/bin/env python3
from typing import Tuple # Not needing in >3.9
import gzip
import pickle
import numpy as np

from data_loader import load_images, load_labels, ascii_image

Array = np.ndarray #Defining a type alias, see PEP 484

#Activation functions

def ReLU(x: Array) -> Array:
    """Applies a very simple ReLU function to a numpy array"""
    return np.maximum(0, x)

def ReLU_derivative(Z: Array) -> Array:
    return Z > 0

def softmax(x: Array) -> Array:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def init_params() -> Tuple[list[Array], list[Array]]:
    """ Generate load model or init values for layers"""
    try:
        with open('params.pkl', 'rb') as f:
            weights, bias = pickle.load(f)
    except FileNotFoundError:
        weights = []
        bias = []
        for i in range(len(layers_topology) - 1):
            weights.append(np.random.rand(layers_topology[i + 1], layers_topology[i]) - 0.5)
            bias.append(np.random.rand(layers_topology[i + 1], 1) - 0.5)
        #Putting values into a list just for management
    return weights, bias

def forward_prop(input_layer: Array, weights: list, bias: list) -> Tuple[list[Array], list[Array]]:
    """Iterate through and activate layers."""
    layers = [input_layer]
    pre_activation_layers = []
    #Loop through hidden layers, leaving just output layer to calculate 
    for w, b in zip(weights[:-1], bias[:-1]):
        pre_activation = np.dot(w, layers[-1]) + b
        pre_activation_layers.append(pre_activation)
        hidden_layer = ReLU(pre_activation)
        layers.append(hidden_layer)
    #Output layer calculation
    pre_activation = np.dot(weights[-1], layers[-1]) + bias[-1]
    output_layer = softmax(pre_activation)
    pre_activation_layers.append(pre_activation)
    layers.append(output_layer)
    return layers, pre_activation_layers

def back_prop(layers: list[Array], pre_activation_layers: list[Array], weights: list[Array], bias: list[Array], n_samples: int, true_y: Array) -> Tuple[list[Array], list[Array]]:
    delta_weights = []
    delta_bias = []
    #Again the output_layer is treated a little differently
    #Detrivative of softmax simplifies to just the differance
    pre_activation_error = layers[-1] - true_y
    delta_w = np.dot(pre_activation_error, layers[-2].T)/n_samples
    delta_b = np.reshape(np.sum(pre_activation_error, 1)/n_samples, (len(layers[-1]), 1))
    delta_weights.append(delta_w)
    delta_bias.append(delta_b)
    
    #-2 to remove input and outlayers
    for i in range(len(layers) - 2, 0, -1):
        pre_activation_error = np.dot(weights[i].T, pre_activation_error) * ReLU_derivative(pre_activation_layers[i-1])
        delta_w = np.dot(pre_activation_error, layers[i-1].T)/n_samples
        delta_b = np.reshape(np.sum(pre_activation_error, 1)/n_samples, (len(layers[i]) , 1))
        delta_weights.append(delta_w)
        delta_bias.append(delta_b)

    #Flip the direction of the lists to the delta for the w1/b1 is first
    return delta_weights[::-1], delta_bias[::-1]

def update_params(weights: list[Array], bias: list[Array], delta_weights: list[Array], delta_bias: list[Array], alpha: float) -> Tuple[list[Array], list[Array]]:
    """ Update weights and bias applying an alpha value/learning value to moderate change"""
    for i in range(len(weights)):
        weights[i] -= alpha * delta_weights[i]
        bias[i] -= alpha * delta_bias[i]
    return weights, bias

def get_predictions(data_to_test: Array, weights: list[Array], bias: list[Array]) -> Array:
    layers, pre_activation_layers = forward_prop(data_to_test, weights, bias)
    return np.argmax(layers[-1], 0)

def get_accuracy(predictions: Array, truth: Array) -> float:
    return np.sum(predictions == truth) / truth.size

def test_accuracy(weights: list[Array], bias: list[Array], i: int, data_to_test: Array, test_labels: Array) -> None:
    predictions = get_predictions(data_to_test, weights, bias)
    print(f"predictions: {predictions[:10]} vs Truth:{test_labels[:10]}")
    accuracy = get_accuracy(predictions, test_labels)
    print(f"Iteration {i} has an avarage accuracy of {accuracy}")
    print("----")

if __name__ == "__main__":
    #Load data and split into train, test, and validation"
    test_images = load_images("MNIST_data/t10k-images-idx3-ubyte.gz", 10000)
    test_labels = load_labels("MNIST_data/t10k-labels-idx1-ubyte.gz", 10000)

    train_images = load_images("MNIST_data/train-images-idx3-ubyte.gz", 60000)
    train_labels = load_labels("MNIST_data/train-labels-idx1-ubyte.gz", 60000)

    #print one image from each set to show it is working
    idx = 123
    ascii_image(train_images[idx], train_labels[idx])

    #Transpose the image so each column is now an image. 
    #Or to say another way, each row is now a single pixel with a value for each image
    #Data is also normalised to stop exp overflow
    train_data = train_images.T/255
    test_data = test_images.T/255

    #Convert a label to an array where the index matching the label is 1, all else are 0
    #I have used enumerate to remind me what this is doing, assignment would be faster
    train_truth = np.zeros((train_labels.size, 10))
    for i, label in enumerate(train_labels):
        train_truth[i, int(label)] = 1
    train_truth = train_truth.T
    print(f"One Hot of image: {train_truth.T[idx]}")

    #Define NN layers

    n_input, n_samples = train_data.shape
    n_hidden = [64, 32, 32] # Add the size for the desired number of neurons per layer
    n_output = 10
    layers_topology = [n_input, *n_hidden, n_output]

    weights, bias = init_params()
    iterations=501
    print("---- Starting training ----")
    for i in range(iterations):
        #Generate outputs using current parms
        layers, pre_activation_layers = forward_prop(train_data, weights, bias)
        #Calculate the error and work backwards
        delta_weights, delta_bias = back_prop(layers, pre_activation_layers, weights, bias, n_samples, train_truth)
        weights, bias = update_params(weights, bias, delta_weights, delta_bias, 0.05)

        if i % 100 == 0:
            # Checking accuracy changes using test data
            test_accuracy(weights, bias, i, test_data, test_labels)

    #Same example image as before        
    test_image = train_images[idx]/255.
    data_to_test = test_image.reshape(784,1)
    prediction = get_predictions(data_to_test, weights, bias)
    print(f"Example image is predicted to be {prediction}")

    #Testing an image I drew in paint
    from PIL import Image
    img = Image.open('2.png').convert('L')
    my_image = np.abs(np.asarray(img).astype(np.float32) - 255)
    print("ascii and prediction of the image I drew")
    ascii_image(my_image, 2)
    data_to_test = my_image.reshape(784, 1)
    prediction = get_predictions(data_to_test, weights, bias)
    print(prediction)

    #Saving params for future
    with open('params.pkl', 'wb') as fw:
        params = pickle.dump([weights, bias], fw)

