from torch import nn
import torch.nn.functional as f

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