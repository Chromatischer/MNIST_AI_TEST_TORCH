import time

import tensorflow as tf
import numpy as np

# Erstellen der Trainingsdaten
inputMuster = tf.constant([[1.0], [2.0], [4.0], [8.0], [10.0]])  # Shape: (3, 1)
outputMuster = tf.constant([[3.0], [6.0], [12.0], [24.0], [30.0]])  # Shape: (3, 1)

# Aufbau des neuronalen Netzwerks
# Initialize weights and bias
W = tf.Variable(tf.random.normal([1, 1]))  # Weight: Shape (1, 1)
b = tf.Variable(tf.random.normal([1]))  # Bias: Shape (1)


# Define the model
def model(x):
    return tf.matmul(x, W) + b


# Loss function (Mean Squared Error)
def loss_function(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


# Optimizer
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# Trainieren des neuronalen Netzwerks
start = time.time()
last_time = start
this_time = start
epochs = 50_000
with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = model(inputMuster)
            loss = loss_function(outputMuster, predictions)

        # Compute gradients and update weights
        gradients = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(gradients, [W, b]))

        last_time = this_time
        this_time = time.time()

        epoch_time = (this_time - last_time) * 1000

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy():.4f}, Time: {epoch_time:.2f} ms")

end = time.time()
training_time = end - start

print("Available devices:", tf.config.list_physical_devices('GPU'))

print(f"Training took: {training_time:.2f} seconds")

# Testen des neuronalen Netzwerks mit Testdaten
testMuster = tf.constant([[22.0], [44.0]])  # Shape: (1, 1)
testPrediction = model(testMuster)
print("Prediction for testMuster 22:", testPrediction.numpy())

print("Model Weights (W):", W.numpy())
print("Model Bias (b):", b.numpy())