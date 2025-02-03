import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Display a sample image
plt.imshow(X_train[2])
plt.show()

# Normalize the dataset
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model
y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Sample prediction
sample_image = X_test[1].reshape(1, 28, 28)
prediction = model.predict(sample_image).argmax(axis=1)
print(f"Predicted Label: {prediction[0]}")

# Display the sample image
plt.imshow(X_test[1])
plt.show()
