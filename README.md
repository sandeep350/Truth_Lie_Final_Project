Truth-Lie Detection Model
This repository contains a deep learning-based model for truth-lie detection using images. The model leverages a Convolutional Neural Network (CNN) to classify images into two categories: Lie and Truth. The project uses TensorFlow and Keras for model development and training.

Table of Contents
Project Overview

Installation

Usage

Model Architecture

Evaluation Metrics

License

Project Overview
This project focuses on training a model that can classify images as either "Lie" or "Truth" based on a given dataset. The dataset consists of labeled images in two classes: Lie and Truth. A CNN architecture is used to train the model, which includes layers for convolution, pooling, flattening, and a dense output layer with a softmax activation function.

Key Features:
Data Preprocessing: Image loading, resizing, and scaling for efficient training.

Model Architecture: Simple CNN with Conv2D, MaxPooling2D layers, followed by a Dense output layer.

Evaluation: Model performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

Hyperparameter Tuning: Integration of Keras Tuner for tuning hyperparameters to optimize model performance.
