#!/bin/env python3
import gzip
import pickle
import numpy as np

from data_loader import load_images, load_labels, ascii_image

#Activation functions

def ReLU(x: np.ndarray) -> np.ndarray:
    """Applies a very simple ReLU function to a numpy array"""
    return np.maximum(0, x)

def ReLU_derivative(Z):
    return Z > 0

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def init_params() -> list:
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

def forward_prop(input_layer: np.ndarray, weights: list, bias: list) -> list:
    """Iterate through and activate layers."""
    layers = [input_layer]
    #Loop through hidden layers, leaving just output layer to calculate 
    for w, b in zip(weights[:-1], bias[:-1]):
        pre_activation = np.dot(w, layers[-1]) + b
        layers.append(pre_activation)
        hidden_layer = ReLU(pre_activation)
        layers.append(hidden_layer)
    #Output layer calculation
    pre_activation = np.dot(weights[-1], layers[-1]) + bias[-1]
    output_layer = softmax(pre_activation)
    layers.append(pre_activation)
    layers.append(output_layer)
    return layers

def back_prop(nn, weights, bias, n_samples, true_y):
    weights_1, weights_2, weights_3 = weights
    bias_1, bias_2, bias_3 = bias
    input_layer, pre_activation_1, hidden_layer_1, pre_activation_2, hidden_layer_2, pre_activation_3, output_layer = nn
    #Detrivative of softmax simplifies to just the differance
    pre_activation_3_error = output_layer - true_y
    delta_weights_3 = np.dot(pre_activation_3_error, hidden_layer_2.T)/n_samples
    delta_bias_3 = np.reshape(np.sum(pre_activation_3_error, 1)/n_samples, (n_output, 1))
    
    pre_activation_2_error = np.dot(weights_3.T, pre_activation_3_error) * ReLU_derivative(pre_activation_2)
    delta_weights_2 = np.dot(pre_activation_2_error, hidden_layer_1.T) / n_samples
    delta_bias_2 = np.reshape(np.sum(pre_activation_2_error, 1)/n_samples, (n_hidden_2, 1))

    pre_activation_1_error = np.dot(weights_2.T, pre_activation_2_error) * ReLU_derivative(pre_activation_1)
    delta_weights_1 = np.dot(pre_activation_1_error, input_layer.T) / n_samples
    delta_bias_1 = np.reshape(np.sum(pre_activation_1_error, 1)/n_samples, (n_hidden_1, 1))
    return [delta_weights_1, delta_bias_1, delta_weights_2, delta_bias_2, delta_weights_3, delta_bias_3]

def update_params(weights, bias, deltas, alpha=0.05):
    delta_weights_1, delta_bias_1, delta_weights_2, delta_bias_2, delta_weights_3, delta_bias_3 = deltas
    weights_1, weights_2, weights_3 = weights
    bias_1, bias_2, bias_3 = bias
    weights_1 -= alpha * delta_weights_1
    weights_2 -= alpha * delta_weights_2
    weights_3 -= alpha * delta_weights_3
    bias_1 -= alpha * delta_bias_1
    bias_2 -= alpha * delta_bias_2
    bias_3 -= alpha * delta_bias_3
    return [[weights_1, weights_2, weights_3], [bias_1, bias_2, bias_3]]

def get_predictions(output_layer):
    return np.argmax(output_layer, 0)

def get_accuracy(predictions, truth):
    return np.sum(predictions == truth) / truth.size

def test_accuracy(weights, bias, i, test_data, test_labels):
    test_output_layer = forward_prop(test_data, weights, bias)[-1]
    predictions = get_predictions(test_output_layer)
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
    n_hidden_1 = 64
    n_hidden_2 = 32
    n_output = 10
    layers_topology = [n_input, n_hidden_1, n_hidden_2, n_output]

    weights, bias = init_params()
    iterations=501
    print("---- Starting training ----")
    for i in range(iterations):
        #Generate outputs using current parms
        nn = forward_prop(train_data, weights, bias)
        #Calculate the error and work backwards
        deltas = back_prop(nn, weights, bias, n_samples, train_truth)
        weights, bias = update_params(weights, bias, deltas)

        if i % 100 == 0:
            # Checking accuracy changes using test data
            test_accuracy(weights, bias, i, test_data, test_labels)

    #Same example image as before        
    test_image = train_images[idx]/255.
    prediction = get_predictions(forward_prop(test_image.reshape(784,1), weights, bias)[-1])
    print(f"Example image is predicted to be {prediction}")

    #Testing an image I drew in paint
    from PIL import Image
    img = Image.open('2.png').convert('L')
    my_image = np.abs(np.asarray(img).astype(np.float32) - 255)
    print("ascii and prediction of the image I drew")
    ascii_image(my_image, 2)
    prediction = get_predictions(forward_prop(my_image.reshape(784, 1), weights, bias)[-1])
    print(prediction)

    #Saving params for future
    with open('params.pkl', 'wb') as fw:
        params = pickle.dump([weights, bias], fw)

