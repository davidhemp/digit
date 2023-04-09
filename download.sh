#!/bin/bash

for f in $(curl "http://yann.lecun.com/exdb/mnist/" | egrep -o "[a-z0-9\-]+\.gz" | sort | uniq); do
    wget -P ./MNIST_data/ "http://yann.lecun.com/exdb/mnist/$f"
done
