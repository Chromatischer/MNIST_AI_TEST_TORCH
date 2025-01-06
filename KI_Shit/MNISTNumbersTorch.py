from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import ssl
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch

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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Verify the data

data_iter = iter(train_loader)
images, labels = next(data_iter)

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

class NeuralNetWithBatchNorm(nn.Module):
    def __init__(self):
        super(NeuralNetWithBatchNorm, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input to first hidden layer
        self.bn1 = nn.BatchNorm1d(128)  # Batch normalization for first hidden layer
        self.fc2 = nn.Linear(128, 64)  # First to second hidden layer
        self.bn2 = nn.BatchNorm1d(64)  # Batch normalization for second hidden layer
        self.fc3 = nn.Linear(64, 10)  # Second hidden to output layer
        self.dropout = nn.Dropout(0.2)  # Dropout layer with 50% rate

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = f.relu(self.bn1(self.fc1(x)))  # First hidden layer with batch norm and ReLU
        x = self.dropout(x)  # Apply dropout
        x = f.relu(self.bn2(self.fc2(x)))  # Second hidden layer with batch norm and ReLU
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)  # Output layer
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

model = NeuralNetWithBatchNorm().to(device)

# Loss function and optimizer

criterion = nn.CrossEntropyLoss()           # Loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

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

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

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

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

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

path = "mnist_model.pth"

torch.save(model.state_dict(), path)
print(f"Saved PyTorch Model Saved to: {path}")