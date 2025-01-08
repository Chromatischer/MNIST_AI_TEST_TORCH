import os.path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import ssl
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch
import time
import threading
import sys

import NeuralNetWithBatchNorm
import pathGen

class ProgressBarThread(threading.Thread):
    def __init__(self, total_epochs, bar_length=50, start_time=time.time()):
        super().__init__()
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        self.time = start_time
        self.bar_length = bar_length
        self.running = True
        self.lock = threading.Lock()

    def update_epoch(self, smthn, total_batches):
        with self.lock:
            self.current_epoch = smthn
            self.total_batches = total_batches
            self.current_batch = 0

    def update_batch(self, batch):
        with self.lock:
            self.current_batch = batch

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            with self.lock:
                # Epoch and batch progress information
                epoch_progress = f"Epoch: {self.current_epoch}/{self.total_epochs}"
                batch_progress = (
                    f" | Batch: {self.current_batch}/{self.total_batches}"
                    if self.total_batches > 0
                    else ""
                )

                # Calculate progress bar
                if self.total_batches > 0:
                    progress = self.current_batch / self.total_batches
                    filled_length = int(self.bar_length * progress)
                    bar = "=" * filled_length + "-" * (self.bar_length - filled_length)
                else:
                    bar = "-" * self.bar_length

                progress_line = f"{epoch_progress} {batch_progress} | [{bar}] {round((time.time() - self.time))}s\r"
                sys.stdout.write(progress_line)
                sys.stdout.flush()
            time.sleep(0.1)  # Update interval

torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

ssl._create_default_https_context = ssl._create_unverified_context
# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoaders

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

# Verify the data

data_iter = iter(train_loader)
images, labels = next(data_iter)

print(images[3])

print(f"Image batch dimensions: {images.shape}")
print(f"Image label dimensions: {labels.shape}")

# Plot the images
fig = plt.figure(figsize=(8, 8))
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(images[i].numpy().squeeze(), cmap='gray')
    ax.set_title(labels[i].item())
# plt.show()

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer (28x28 pixels)
        self.fc2 = nn.Linear(128, 64)       # Hidden layer
        self.fc3 = nn.Linear(64, 10)        # Output layer (10 classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = f.relu(self.fc1(x))  # Apply ReLU activation
        x = f.relu(self.fc2(x))  # Another ReLU
        x = self.fc3(x)          # Output layer
        return x

class NeuralNetWithDropout(nn.Module):
    def __init__(self):
        super(NeuralNetWithDropout, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input to first hidden layer
        self.fc2 = nn.Linear(128, 64)       # First to second hidden layer
        self.fc3 = nn.Linear(64, 10)        # Second hidden to output layer
        self.dropout = nn.Dropout(0.2)     # Dropout layer with 50% rate -> changed to 20%

    def forward(self, x):
        x = x.view(-1, 28 * 28)           # Flatten the image
        x = f.relu(self.fc1(x))           # First hidden layer
        x = self.dropout(x)               # Apply dropout
        x = f.relu(self.fc2(x))           # Second hidden layer
        x = self.dropout(x)               # Apply dropout
        x = self.fc3(x)                   # Output layer
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max-pooling with 2x2 window

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Flattened input size
        self.fc2 = nn.Linear(128, 10)         # Output layer
        self.dropout = nn.Dropout(0.2)       # Dropout for regularization

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = f.relu(self.conv1(x))
        x = self.pool(f.relu(self.conv2(x)))

        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        x = f.relu(self.fc1(x))  # First fully connected layer
        x = self.dropout(x)      # Apply dropout
        x = self.fc2(x)          # Output layer
        return x

model = NeuralNetWithBatchNorm.NeuralNetWithBatchNorm().to(device)

# Loss function and optimizer

criterion = nn.CrossEntropyLoss()           # Loss function
optimizer = optim.AdamW(model.parameters(), lr=0.005)  # Optimizer

for images, labels in train_loader:
    # Debugging shapes
    print(f"Images shape: {images.shape}")  # Expected: [batch_size, 1, 28, 28]
    print(f"Labels shape: {labels.shape}")  # Expected: [batch_size]

    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)  # Ensure batch sizes match here
    loss.backward()
    optimizer.step()
    break  # Test with one batch to debug
# Training the model

epochs = 15

start = time.time()
progress_thread = ProgressBarThread(total_epochs=epochs)
progress_thread.start()
last = start
curr = start

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    progress_thread.update_epoch(epoch + 1, len(train_loader))

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        last = curr
        curr = time.time()
        progress_thread.update_batch(progress_thread.current_batch + 1)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Time: {(curr - last) * 1000:.2f} ms")

progress_thread.stop()
progress_thread.join()

print(f"Training took: {time.time() - start:.2f} seconds")

# Test the model

model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
# Best: 93.99% using NeuralNetWithDropout 69.4% using NeuralNet 96.45% using NeuralNetWithDropout with 20% dropout
# 97.3% using 10 epochs 98.1% 10 epochs and BatchNormalization 98.3% using 15 epochs and BatchNormalization
#

# Save the model

path = pathGen.generate_unique_filename("mnist_model.pth", "model-Iterations")

torch.save(model.state_dict(), path)
print(f"Saved PyTorch Model Saved to: {path}")