!/usr/bin/env bash

# script to preprocess data

echo "Downloading CIFAR10..."
cd ./cifar10/preprocessing
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz

echo "Extracting images for pickle files..."

mkdir -p ../data/img/train
mkdir -p ../data/img/test

python save_images.py

echo "Downloading CIFAR100..."
cd ./cifar100/preprocessing
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvf cifar-100-python.tar.gz
rm cifar-100-python.tar.gz

echo "Extracting images for pickle files..."
mkdir -p ../data/img/train
mkdir -p ../data/img/test
python save_images.py
