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

