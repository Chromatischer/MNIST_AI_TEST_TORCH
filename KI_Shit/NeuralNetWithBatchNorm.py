from torch import nn
import torch.nn.functional as f

class NeuralNetWithBatchNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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

def printCleanNumpyArray(arr):
    # Print the array with two decimal places and keeping the lines cleen
    for x in arr:
        for y in x:
            # remove the negative sign and change color to read instead
            print(f"{y:.2f}", end=" ")
        print()
    print()

def number_Modifier(num):
    return abs(max(num, 0) * 10)