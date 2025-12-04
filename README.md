# CIFAR-10 Image Classifier

This project implements a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset using PyTorch. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), with 6,000 images per class.

## Project Structure

- `CIFAR-10_Image_Classifier.ipynb`: Jupyter notebook containing the complete implementation, including data loading, model definition, training, and evaluation.
- `cifar_net.pth`: Saved trained model weights.
- `data/`: Directory containing the CIFAR-10 dataset (downloaded automatically if not present).

## Requirements

- Python 3.8+
- PyTorch
- Torchvision
- Matplotlib
- NumPy

## Setup

Create a conda environment with the required dependencies:

```bash
conda create -n cifar10 python=3.8 pytorch torchvision matplotlib numpy -c pytorch
conda activate cifar10
```

## Usage

1. Open the `CIFAR-10_Image_Classifier.ipynb` notebook.
2. Run the cells in order to load data, define the model, train, and evaluate.
3. The trained model weights are saved as `cifar_net.pth`.

## Model Architecture

The CNN consists of convolutional layers followed by fully connected layers, using ReLU activations and max pooling for feature extraction.