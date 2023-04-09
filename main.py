#!/bin/env python3
import gzip
import numpy as np

def load_images(filename: str, num_images: int) -> np.ndarray:
    with gzip.open(filename, 'r') as f:
        image_size = 28
        f.read(16)
        buffer = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buffer,  dtype=np.uint8).astype(np.float32)
        data =  data.reshape(num_images, image_size, image_size, 1)
    return data

def load_labels(filename: str, num_labels: int) -> np.ndarray:
    with gzip.open(filename, 'r') as f:
        f.read(8)
        buffer = f.read(num_labels)
        labels = np.frombuffer(buffer,  dtype=np.uint8).astype(np.float32)
    return labels

def ascii_image(image: np.ndarray, label: np.float32):
    print(f"Image labeled as {label}")
    for line in image:
        print("".join([ "{:03d}".format(int(i)) for i in line]))

test_images = load_images("MNIST_data/t10k-images-idx3-ubyte.gz", 10000)
test_labels = load_labels("MNIST_data/t10k-labels-idx1-ubyte.gz", 10000)

test_images = load_images("MNIST_data/train-images-idx3-ubyte.gz", 10000)
test_labels = load_labels("MNIST_data/train-labels-idx1-ubyte.gz", 10000)

idx = 666
ascii_image(test_images[idx], test_labels[idx])
ascii_image(test_images[idx], test_labels[idx])

