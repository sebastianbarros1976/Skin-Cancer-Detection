# Skin Cancer Detection with CNN and Gradio

This project demonstrates the use of a Convolutional Neural Network (CNN) to classify images of skin lesions as benign or malignant. The project utilizes PyTorch for model training and Gradio for creating a user-friendly interface for predictions.

## Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Data Preparation](#data-preparation)
5. [Training the Model](#training-the-model)
6. [Evaluation](#evaluation)
7. [Creating a Gradio Interface](#creating-a-gradio-interface)
8. [Running the Application](#running-the-application)
9. [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Skin-Cancer-Detection-with-CNN-and-Gradio.git
    cd Skin-Cancer-Detection-with-CNN-and-Gradio
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset used for this project is the [Skin Cancer: Malignant vs. Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign/data) dataset from Kaggle.

1. Download the dataset from Kaggle and extract it into a directory named `data`.

    ```bash
    kaggle datasets download -d fanconic/skin-cancer-malignant-vs-benign
    unzip skin-cancer-malignant-vs-benign.zip -d data
    ```

## Model Architecture

The model used in this project is a Convolutional Neural Network (CNN) with the following architecture:

- 4 Convolutional layers with batch normalization and max pooling
- 2 Fully connected layers
- Dropout for regularization

## Data Preparation

Data preparation includes transforming the images and splitting the dataset into training, validation, and test sets.

## Training the Model

Training the model involves several epochs of forward and backward propagation using a defined loss function and optimizer.

## Evaluation

Evaluation includes checking the model's performance on the validation set and using metrics such as accuracy, confusion matrix, and classification report.

## Creating a Gradio Interface

We use Gradio to create an interface for users to upload images and get predictions.

## Running the Application

Run the following command to start the Gradio interface:
```bash
python main.py
