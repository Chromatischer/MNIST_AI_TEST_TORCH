from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import ssl
import torch.nn as nn
import torch.optim as optim
import torch
import time

import CNN_Network
import NeuralNetWithBatchNorm
import pathGen

EXTENDED_DEBUG = False                # Print extended debug information
DO_SAVE_MODEL = True                  # Save by default
SAVE_MODEL_ABOVE_ACCURACY = 98.2      # Save if accuracy is above this value even if DO_SAVE_MODEL is False
ENABLE_CUDNN_BENCHMARK = True         # Enable CUDNN Benchmarking
LOADER_BATCH_SIZE = 512               # Batch size for the DataLoader
ALLOW_DATA_DOWNLOAD = False           # Allow downloading the data
TRAIN_EPOCHS = 25                     # Number of epochs to train for

torch.backends.cudnn.benchmark = ENABLE_CUDNN_BENCHMARK
print("CUDNN Benchmarking: ", "Enabled" if ENABLE_CUDNN_BENCHMARK else "Disabled")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.RandomRotation(10),  # Rotate the image by 10 degrees
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

ssl._create_default_https_context = ssl._create_unverified_context # Fix for SSL error

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=ALLOW_DATA_DOWNLOAD, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=ALLOW_DATA_DOWNLOAD, transform=transform)

# Create DataLoaders

train_loader = DataLoader(train_dataset, batch_size=LOADER_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=LOADER_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# Verify the data

data_iter = iter(train_loader)
images, labels = next(data_iter)

print("Data Loaded!")

if EXTENDED_DEBUG: print(images[3])

if EXTENDED_DEBUG: print(f"Image batch dimensions: {images.shape}")
if EXTENDED_DEBUG: print(f"Image label dimensions: {labels.shape}")

# Plot the images
fig = plt.figure(figsize=(8, 8))
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(images[i].numpy().squeeze(), cmap='gray')
    ax.set_title(labels[i].item())
if EXTENDED_DEBUG: plt.show()

model = CNN_Network.CNNForMNIST().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()           # Loss function
optimizer = optim.AdamW(model.parameters(), lr=0.01)  # Optimizer

# Debugging shapes
if EXTENDED_DEBUG:
    for images, labels in train_loader:
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
start = time.time()
progress_thread = NeuralNetWithBatchNorm.ProgressBarThread(total_epochs=TRAIN_EPOCHS)
progress_thread.start()
last = start
curr = start

for epoch in range(TRAIN_EPOCHS):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    progress_thread.update_epoch(epoch + 1, len(train_loader))
    progress_thread.update_estimate(curr - last)

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

        progress_thread.update_batch(progress_thread.current_batch + 1)

    last = curr
    curr = time.time()

    print(f"Epoch {epoch + 1}/{TRAIN_EPOCHS}, Loss: {running_loss / len(train_loader):.4f}, Took: {(curr - last):.2f}s")

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
# 20minutes of fucking training and using CNN gives accuracy of 98.7% with 20 epochs, saved as 21.pth
# The CNN takes 60s per Epoch...

# Save the model
if DO_SAVE_MODEL or (0 < SAVE_MODEL_ABOVE_ACCURACY < 100 * correct / total):
    path = pathGen.generate_unique_filename("mnist_model.pth", "model-Iterations")

    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model Saved to: {path}")
else:
    print("Model not saved due to config!")