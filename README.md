**MNIST Digit Classification with TensorFlow & Keras**
**Overview**
This project implements a simple neural network for classifying handwritten digits from the MNIST dataset using TensorFlow and Keras. The model is a fully connected feedforward neural network trained on the dataset and evaluated for accuracy.

**Dataset**
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. It is split into:

Training Set: 60,000 images
Test Set: 10,000 images

**Model Architecture**
The model is a simple deep neural network with the following layers:

Flatten Layer: Converts 28x28 images into a 1D array of 784 values.

Dense Layer (128 neurons, ReLU activation): Learns high-level features.

Dense Layer (32 neurons, ReLU activation): Further abstraction of features.

Dense Layer (10 neurons, Softmax activation): Outputs probabilities for 10 digit classes (0-9).

**Installation**
To run this project, ensure you have Python installed along with the required libraries:
pip install tensorflow matplotlib scikit-learn
**Running the Code**
Load and preprocess the MNIST dataset.
Define and compile the neural network.
Train the model on the dataset.
Evaluate the model on the test set.
Predict and visualize results.

To run the script, simply execute:
python mnist_classification.py

**Performance**
The model is trained for 10 epochs with Adam optimizer and achieves a reasonable accuracy on the test set. The accuracy can be improved with further tuning, data augmentation, or convolutional layers (CNNs).

**Results**
The trained model achieves a decent accuracy in classifying handwritten digits. Further optimizations like dropout layers, batch normalization, or CNNs can enhance performance.
