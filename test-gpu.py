#!/bin/env python3
import tensorflow as tf

gpus=tf.config.list_physical_devices("GPU")
print(f"GPUs Available: {gpus}")
