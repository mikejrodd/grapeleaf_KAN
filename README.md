# Grape Leaf Disease Detection with KANConv2D

## Description
This project uses a custom convolutional layer, KANConv2D, to build a convolutional neural network (CNN) for detecting Esca disease in grape leaves. The KANConv2D layer incorporates kernel adaptive networks to enhance the performance of traditional Conv2D layers.

## Features
- Custom KANConv2D layer with kernel adaptive networks
- Data augmentation for robust training
- Early stopping to prevent overfitting
- Class weighting to handle imbalanced datasets

## Data Preparation
The data preparation involves organizing the grape leaf images into training and testing directories, each containing subdirectories for healthy and Esca-affected leaves.

## Data Augmentation
The training data is augmented using the following transformations:
- Rescaling
- Rotation
- Width and height shifts
- Shear and zoom
- Horizontal flip

## Model Architecture
The CNN model consists of:
- Multiple KANConv2D layers
- Max pooling layers
- Dense and dropout layers
- Binary output with sigmoid activation

## Training
The model is compiled with the Adam optimizer, binary cross-entropy loss, and accuracy as the metric. Class weights are adjusted to reduce false negatives for Esca detection. The model is trained with early stopping to restore the best weights based on validation loss.

## Installation
To run this program, you need to have TensorFlow installed. You can install it using pip:

```bash
pip install tensorflow
```

## Usage
1. Prepare the data by organizing the grape leaf images into training and testing directories.
2. Run the script to train the model and save the best model to disk.
3. Evaluate the model on the validation dataset.

## Results
The trained model achieves a certain level of accuracy and loss on the validation dataset. The results are printed at the end of the training script.

## License
This project is licensed under the MIT License.
