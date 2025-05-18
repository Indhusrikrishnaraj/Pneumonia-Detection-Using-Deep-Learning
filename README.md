# Pneumonia Detection from Chest X-Ray Images using CNN

This project leverages deep learning to detect pneumonia from chest X-ray images. Using convolutional neural networks (CNNs), the model classifies images as either *Pneumonia* or *Normal*. The implementation includes both a base CNN model and an optimized version using hyperparameter tuning with Optuna.

##  Model Overview

The model is a multi-layer CNN built with TensorFlow and Keras. It uses the following features:
- Data Augmentation using `ImageDataGenerator`
- Custom CNN architecture with Conv2D and MaxPooling2D layers
- EarlyStopping and ModelCheckpoint callbacks
- Performance metrics: Accuracy, Precision, Recall
- Hyperparameter tuning using Optuna for model optimization

##  Dataset

The dataset used is the [Chest X-ray Images (Pneumonia) dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia), which contains X-ray images organized into training, validation, and test folders with `NORMAL` and `PNEUMONIA` subdirectories.

### Class Distribution
- Training: 4434 images
- Validation: 782 images
- Testing: 624 images


##  Training Strategy

The model is trained for up to 20 epochs with early stopping. The hyperparameters optimized include:
- Number of convolutional layers
- Number of filters and kernel size
- Pooling strategy (max/average)
- Dense layer units
- Dropout rate
- Learning rate

##  Performance

### Final CNN Model (After Hyperparameter Tuning):
- **Accuracy:** 85.90%
- **Precision:** 84.32%
- **Recall:** 95.13%
- **Loss:** 0.36 on test set

### Evaluation Metrics Tracked:
- Loss
- Accuracy
- Precision
- Recall

##  Visualizations

The project includes:
- Class distribution bar charts
- Training vs Validation Loss, Accuracy, Precision, Recall plots
- Residual plots for linear model comparisons (optional)

##  Requirements

Install dependencies with:

```bash
pip install -r requirements.txt



