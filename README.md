# Facial-Expression-Recognition-Model
This project implements a Facial Expression Recognition (FER) system using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset. The model classifies human facial expressions into emotion categories

Built and trained on Kaggle, this project aims to demonstrate:

How CNNs can learn to recognize facial emotions.

How explainable AI can increase trust in computer vision models.

The role of facial emotion recognition in decision-making systems.

## Objectives
Train a CNN on FER-2013 to classify facial expressions.

Achieve robust performance through 55 training epochs.

Lay the foundation for adding explainable AI techniques (e.g., Grad-CAM).

Provide a stepping stone toward a Manifested AI approach—models that are accurate, interpretable, and ethically aligned.

## Dataset
FER-2013

Source: Kaggle FER-2013 Dataset

Categories: Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral

Images: 48x48 grayscale face images in CSV format.

Split: Training / Public Test / Private Test sets.

## Tech Stack
Python 3.10+

TensorFlow / Keras – Model building and training

OpenCV – Image preprocessing (optional)

Matplotlib / Seaborn – Visualization

NumPy / Pandas – Data handling

ImageDataGenerator

## Model Architecture
Input Layer: 48x48 grayscale images

Convolutional Layers: Multiple Conv2D + ReLU + BatchNorm layers

Pooling Layers: MaxPooling2D to reduce spatial dimensions

Fully Connected Layers: Dense layers with Dropout for regularization

Output Layer: Softmax activation for 7 emotion categories

## Training
Epochs: 55

Batch Size: 64

Optimizer: Adam (learning rate = 0.001)

Loss Function: Categorical Crossentropy

Augmentation: Random rotation, zoom, horizontal flip


## How to Run on Kaggle
Open the Kaggle Notebook.

Attach the FER-2013 dataset to your notebook.

Install required packages (if not pre-installed).

Run all cells to train and evaluate the model.


## Future Improvements
Add Grad-CAM explainability module.

Experiment with Transfer Learning (e.g., ResNet-50).

Reduce bias by evaluating performance across demographic groups.

Deploy as a web app using Streamlit or Gradio.

## License
MIT License – feel free to use and adapt with attribution.
