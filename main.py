#!/bin/env python3
import gzip
import numpy as np
from data_loader import load_images, load_labels, ascii_image

#Load data and split into train, test, and validation"
test_images = load_images("MNIST_data/t10k-images-idx3-ubyte.gz", 10000)
test_labels = load_labels("MNIST_data/t10k-labels-idx1-ubyte.gz", 10000)

train_images = load_images("MNIST_data/train-images-idx3-ubyte.gz", 60000)
train_labels = load_labels("MNIST_data/train-labels-idx1-ubyte.gz", 60000)

train_images, validation_images = train_images[:55000], train_images[55000:]
train_labels, validation_labels = train_labels[:55000], train_labels[55000:]

#print one image from each set to show it is working
debug = 0
if debug:
    idx = 123
    ascii_image(test_images[idx], test_labels[idx])
    ascii_image(train_images[idx], train_labels[idx])
    ascii_image(validation_images[idx], validation_labels[idx])



