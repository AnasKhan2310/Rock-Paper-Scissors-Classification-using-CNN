# Rock-Paper-Scissors-Classification-using-CNN
A deep learning image classification project built using TensorFlow/Keras, trained on the Rock–Paper–Scissors dataset from TensorFlow Datasets (TFDS).

Project Overview
This project demonstrates how Convolutional Neural Networks (CNNs) can be used for real-world image classification.
The model classifies hand gesture images into three categories:

✊ Rock

✋ Paper

✌️ Scissors

The complete pipeline includes:

Dataset loading

Image preprocessing

Model building

Training

Evaluation

Visualization

 Dataset

Dataset: Rock–Paper–Scissors
Source: TensorFlow Datasets (TFDS)

import tensorflow_datasets as tfds
dataset, info = tfds.load("rock_paper_scissors", as_supervised=True, with_info=True)

🧾 Model Architecture

The CNN consists of:

Conv2D (32 filters) → ReLU

Conv2D (64 filters) → ReLU

Conv2D (128 filters) → ReLU

Flatten Layer

Dense (128 neurons) → ReLU

Dense (3 neurons) → Softmax

Loss Function: sparse_categorical_crossentropy
Optimizer: adam

🚀 Training
history = model.fit(
    train,
    epochs=10,
    validation_data=test
)

📊 Performance Visualization

The notebook includes:

Accuracy Curve

Loss Curve

These help understand training behavior and model generalization.

🎯 Key Concepts Explained

Why ReLU is used in hidden layers

Why Softmax is ideal for multi-class output

Why Categorical Crossentropy is required

How to reduce overfitting
