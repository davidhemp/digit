#!/bin/env python3
import gzip
import pickle
import numpy as np

from data_loader import load_images, load_labels, ascii_image

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

n_input = train_data.shape[0]
n_samples = train_data.shape[1]
n_hidden = 10
n_output = 10

#Activation functions

def ReLU(x: np.ndarray) -> np.ndarray:
    """Applies a very simple ReLU activation function to a numpy array"""
    return np.maximum(0, x)

def ReLU_derivative(Z):
    return Z > 0

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Generate init values for layers

try:
    with open('params.pkl', 'rb') as f:
        params = pickle.load(f)
except FileNotFoundError:
    weights_1 = np.random.rand(n_hidden, n_input) - 0.5
    bias_1 = np.random.rand(n_hidden, 1) - 0.5

    weights_2 = np.random.randn(n_output, n_output) - 0.5
    bias_2 = np.random.randn(n_output, 1) - 0.5
    #Putting values into a list just for management
    params = [weights_1, bias_1, weights_2, bias_2]

# Apply weights and bias then activate
def forward_prop(input_layer, params):
    weights_1, bias_1, weights_2, bias_2 = params
    a_1 = np.dot(weights_1, input_layer)
    a_2 = a_1 + bias_1
    a_3 = ReLU(a_2)
    pre_activation_1 = np.dot(weights_1, input_layer) + bias_1
    hidden_layer_1 = ReLU(pre_activation_1)
    pre_activation_2 = np.dot(weights_2, hidden_layer_1) + bias_2
    output_layer = softmax(pre_activation_2)
    return [input_layer, pre_activation_1, hidden_layer_1, pre_activation_2, output_layer]

def back_prop(nn, params, n_samples, true_y):
    weights_1, bias_1, weights_2, bias_2 = params
    input_layer, pre_activation_1, hidden_layer_1, pre_activation_2, output_layer = nn
    #Detrivative of softmax simplifies to just the differance
    pre_activation_2_error = output_layer - true_y
    delta_weights_2 = np.dot(pre_activation_2_error, hidden_layer_1.T)/n_samples
    delta_bias_2 = np.sum(pre_activation_2_error, 1)/n_samples
    
    pre_activation_error_1 = np.dot(weights_2, pre_activation_2_error) * ReLU_derivative(pre_activation_1)
    delta_weights_1 = np.dot(pre_activation_error_1, input_layer.T) / n_samples
    delta_bias_1 = np.sum(pre_activation_error_1, 1)/n_samples
    return [delta_weights_1, delta_bias_1, delta_weights_2, delta_bias_2]

def update_params(params, deltas, alpha=0.1):
    delta_weights_1, delta_bias_1, delta_weights_2, delta_bias_2 = deltas
    weights_1, bias_1, weights_2, bias_2 = params

    weights_1 -= alpha * delta_weights_1
    weights_2 -= alpha * delta_weights_2
    bias_1 -= alpha * np.reshape(delta_bias_1, (10,1))
    bias_2 -= alpha * np.reshape(delta_bias_2, (10,1))
    
    return [weights_1, bias_1, weights_2, bias_2]

def get_predictions(output_layer):
    return np.argmax(output_layer, 0)

def get_accuracy(predictions, truth):
    return np.sum(predictions == truth) / truth.size

def test_accuracy(params, i, test_data, test_labels):
    test_output_layer = forward_prop(test_data, params)[4]
    predictions = get_predictions(test_output_layer)
    print(f"predictions: {predictions[:10]} vs Truth:{test_labels[:10]}")
    accuracy = get_accuracy(predictions, test_labels)
    print(f"Iteration {i} has an avarage accuracy of {accuracy}")
    print("----")

iterations=501
print("---- Starting training ----")
for i in range(iterations):
    #Generate outputs using current parms
    nn = forward_prop(train_data, params)
    #Calculate the error and work backwards
    deltas = back_prop(nn, params, n_samples, train_truth)
    params = update_params(params, deltas)

    if i % 100 == 0:
        # Checking accuracy changes using test data
        test_accuracy(params, i, test_data, test_labels)

#Same example image as before        
test_image = train_images[idx]/255.
prediction = get_predictions(forward_prop(test_image.reshape(784,1), params)[4])
print(f"Example image is predicted to be {prediction}")

#Testing an image I drew in paint
from PIL import Image
img = Image.open('2.png').convert('L')
my_image = np.abs(np.asarray(img).astype(np.float32) - 255)
print("ascii and prediction of the image I drew")
ascii_image(my_image, 2)
prediction = get_predictions(forward_prop(my_image.reshape(784, 1), params)[4])
print(prediction)

#Saving params for future
with open('params.pkl', 'wb') as fw:
    params = pickle.dump(params, fw)

