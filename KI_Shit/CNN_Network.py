import torch
import torch.nn as nn
import torch.nn.functional as f

class CNNForMNIST(nn.Module):
    def __init__(self):
        super(CNNForMNIST, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Output: 32x28x28
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64x28x28
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions by half

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Flattened after pooling
        self.fc2 = nn.Linear(128, 10)  # Output for 10 classes (digits 0-9)
        self.dropout = nn.Dropout(0.25)  # Dropout for regularization

    def forward(self, x):
        # Convolutional layers with ReLU and BatchNorm
        x = f.relu(self.bn1(self.conv1(x)))  # Conv1 -> BN -> ReLU
        x = f.relu(self.bn2(self.conv2(x)))  # Conv2 -> BN -> ReLU
        x = self.pool(x)  # MaxPooling

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = f.relu(self.fc1(x))  # Fully connected layer 1
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Fully connected layer 2 (output layer)
        return x