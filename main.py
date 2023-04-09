#!/bin/env python3
import gzip
import numpy as np

def load_images(filename: str, num_images: int) -> np.ndarray:
    """ Takes input file from MNIST dataset and returns flat array for each image"""
    with gzip.open(filename, 'r') as f:
        image_size = 28
        f.read(16)
        buffer = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buffer,  dtype=np.uint8).astype(np.float32)
        data =  data.reshape(num_images, image_size * image_size)
    return data

def load_labels(filename: str, num_labels: int) -> np.ndarray:
    """ Takes input file from MNIST dataset and returns a numpy array of labels"""
    with gzip.open(filename, 'r') as f:
        f.read(8)
        buffer = f.read(num_labels)
        labels = np.frombuffer(buffer,  dtype=np.uint8).astype(np.float32)
    return labels

def ascii_image(data: np.ndarray, label: np.float32):
    """prints an ascii image of the flattened array from load_images"""
    image_size = 28
    image = data.reshape(image_size, image_size)
    print(f"Image labeled as {label}")
    for line in image:
        print("".join([ "{:03d}".format(int(i)) for i in line]))

#Load data and print one image from each set to show it is working
test_images = load_images("MNIST_data/t10k-images-idx3-ubyte.gz", 10000)
test_labels = load_labels("MNIST_data/t10k-labels-idx1-ubyte.gz", 10000)

train_images = load_images("MNIST_data/train-images-idx3-ubyte.gz", 60000)
train_labels = load_labels("MNIST_data/train-labels-idx1-ubyte.gz", 60000)

train_images, validation_images = train_images[:55000], train_images[55000:]
train_labels, validation_labels = train_labels[:55000], train_labels[55000:]

idx = 123
ascii_image(test_images[idx], test_labels[idx])
ascii_image(train_images[idx], train_labels[idx])
ascii_image(validation_images[idx], validation_labels[idx])



