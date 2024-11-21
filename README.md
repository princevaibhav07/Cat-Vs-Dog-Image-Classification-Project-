# Cat Vs Dog Image Classification Project | Deep Learning Project | CNN Project

## Project Overview

This project focuses on building a Convolutional Neural Network (CNN) model to classify images of dogs and cats. The dataset was sourced from Kaggle, specifically the "Dogs vs Cats" dataset by Salader. The model was trained on labeled images of dogs and cats, with the goal of achieving high accuracy in distinguishing between the two classes.

The project demonstrates the complete pipeline of downloading the dataset, preprocessing the data, training the CNN model, and evaluating its performance on unseen data.

### Project Features

Dataset: The Kaggle "Dogs vs Cats" dataset, downloaded and processed.

### Preprocessing:

Image resizing to 256x256.

Normalization of pixel values to the range [0,1].

### Model Architecture:

A sequential CNN model with multiple convolutional layers, batch normalization, dropout layers, and dense layers.

The model uses ReLU activation functions for intermediate layers and a sigmoid activation function for binary classification.

### Training Details:

Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy

Validation and Testing:

Validation set performance was monitored at each epoch.

Predictions were made on test images to showcase the model's performance.

## Accuracy and Loss Details

Below are the accuracy and loss metrics achieved during training and validation across 10 epochs:

| **Epoch** | **Training Accuracy** | **Training Loss** | **Validation Accuracy** | **Validation Loss** |
|-----------|------------------------|--------------------|--------------------------|----------------------|
| 1         | 55.46%                | 2.7899             | 64.14%                  | 0.6315              |
| 2         | 64.93%                | 0.6423             | 66.26%                  | 0.6449              |
| 3         | 71.32%                | 0.5557             | 76.82%                  | 0.4827              |
| 4         | 77.02%                | 0.4748             | 77.82%                  | 0.4686              |
| 5         | 79.65%                | 0.4307             | 73.56%                  | 0.6152              |
| 6         | 82.01%                | 0.3841             | 77.88%                  | 0.4645              |
| 7         | 85.06%                | 0.3310             | 79.32%                  | 0.4386              |
| 8         | 87.63%                | 0.2667             | 78.78%                  | 0.5584              |
| 9         | 90.58%                | 0.2023             | 81.32%                  | 0.4405              |
| 10        | 92.64%                | 0.1641             | 84.80%                  | 0.4692              |


## Model Architecture

The CNN model includes the following components:

Convolutional Layers: Extract spatial features from images.

Batch Normalization: Standardize the inputs for faster convergence.

MaxPooling Layers: Reduce the spatial dimensions of feature maps.

Dense Layers: Learn complex relationships between features.

Dropout: Prevent overfitting by randomly dropping nodes during training.

### Summary of Model:

Input Shape: (256, 256, 3)

Total Parameters: Trainable (large-scale), suitable for GPUs.

Last Layer: Sigmoid activation for binary classification (0 for cat, 1 for dog).

#### Visualization of Results

Training and validation accuracy and loss were plotted for better understanding of the model's learning progress:

### Accuracy Graph:

Demonstrates the improvement of training and validation accuracy across epochs.

#### Loss Graph:

Highlights the reduction in training and validation loss across epochs.

### Ways to Reduce Overfitting

Data Augmentation, 
L1/L2 Regularization, 
Dropout Layers, 
Batch Normalization, 
Adding more data, 
Reducing the complexity of the model architecture, 

Predicting on Test Images, 
#### Two test images (cat.jpeg and dog.jpeg) were used to test the model's performance on unseen data:

Images were resized to 256x256.

Predictions were generated using the trained model.

###Results:

cat.jpeg: Predicted as "Cat" with high confidence.

dog.jpeg: Predicted as "Dog" with high confidence.
