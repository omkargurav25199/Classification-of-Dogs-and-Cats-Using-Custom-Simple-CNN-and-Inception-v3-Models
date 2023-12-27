# Classification-of-Dogs-and-Cats-Using-Custom-Simple-CNN-and-Inception-v3-Models with TensorFlow and Pytorch


## TensorFlow Image Classification

## Introduction
This repository demonstrates two approaches to image classification using TensorFlow: one with a custom Convolutional Neural Network (CNN) and the other utilizing the Inception V3 pre-trained model. Both models are trained on a dataset of cats and dogs for binary classification.

## Dataset
The dataset is located in the directory 'kagglecatsanddogs_5340\PetImages'. It contains images of cats and dogs for training and validation.

## Custom CNN Model
### Data Preprocessing
Images are loaded and preprocessed using the `ImageDataGenerator` from Keras, including rescaling, shear range, zoom range, and horizontal flip. The dataset is split into training and validation sets.

### Model Architecture
A custom CNN model is designed with layers for feature extraction and classification. It includes convolutional layers, max-pooling layers, and dense layers. The model is compiled with binary cross-entropy loss and the Adam optimizer.

### Model Training
The CNN model is trained on the training set using the `fit_generator` function. After training for 10 epochs, the model achieves an accuracy of approximately 91.78% on the training set and 88.55% on the validation set.

## Inception V3 Model
### Data Preprocessing
Images are loaded and preprocessed using the `ImageDataGenerator` from Keras, including rescaling and horizontal flip. The dataset is split into training and validation sets.

### Model Architecture
The Inception V3 model is loaded with pre-trained weights (excluding the top layer) from 'imagenet'. A custom dense layer is added for binary classification. The model is then compiled with the Adam optimizer and binary cross-entropy loss.

### Model Training
The Inception V3 model is trained on the training set using the `fit` function. After training for 10 epochs, the model achieves an accuracy of approximately 96.14% on the validation set.

## Usage
1. Clone the repository.
2. Ensure TensorFlow is installed (`pip install tensorflow`).
3. Run the provided Python scripts for training and evaluation for both the custom CNN model and the Inception V3 model.

## Model Summaries
Detailed summaries of the model architectures and parameters are provided using the `model.summary()` function.

## Adjustments
- Fine-tune hyperparameters such as learning rate, batch size, and the number of epochs based on your preferences and dataset size.
- Modify the model architecture or include additional layers if needed for your specific task.



## PyTorch Image Classification

This repository showcases two image classification models implemented in PyTorch: one using a custom Convolutional Neural Network (CNN) and the other leveraging the Inception V3 pre-trained model. Both models are trained on a dataset of cats and dogs for binary classification.

## Custom CNN Model
### Data Preprocessing
Images are loaded and preprocessed using PyTorch's `transforms` module, including resizing and converting to tensors. The dataset is split into training and validation sets.

### Model Architecture
A custom CNN model (`SimpleCNN`) is defined with convolutional layers, max-pooling layers, and fully connected layers. The model is trained using binary cross-entropy loss and the Adam optimizer.

### Model Training
The CNN model is trained for 10 epochs on the training set, achieving an accuracy of approximately 84.24% on the validation set.

## Inception V3 Model
### Data Preprocessing
Similar to the CNN model, images are loaded and preprocessed using PyTorch's `transforms`. The dataset is also split into training and validation sets.

### Model Architecture
The Inception V3 model is loaded with pre-trained weights from 'imagenet', and its output layer is modified for binary classification. The model is trained using binary cross-entropy with logits loss and the SGD optimizer.

### Model Training
The Inception V3 model is trained for 5 epochs on the training set, achieving an accuracy of approximately 94.32% on the validation set.

## Usage
1. Clone the repository.
2. Ensure PyTorch is installed (`pip install torch`).
3. Run the provided Python scripts for training and evaluation for both the custom CNN model and the Inception V3 model.

## Model Summaries
Detailed summaries of the model architectures and parameters are provided using the appropriate PyTorch functions.

## Adjustments
- Fine-tune hyperparameters such as learning rate, batch size, and the number of epochs based on your preferences and dataset size.
- Modify the model architecture or include additional layers if needed for your specific task.
