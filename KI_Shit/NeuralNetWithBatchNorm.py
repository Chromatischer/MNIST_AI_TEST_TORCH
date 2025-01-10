from torch import nn
import torch.nn.functional as f
import threading
import time
import sys

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

class ProgressBarThread(threading.Thread):
    def __init__(self, total_epochs, bar_length=50, start_time=time.time()):
        super().__init__()
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        self.time = start_time
        self.estimated_remaining = 0
        self.bar_length = bar_length
        self.running = True
        self.lock = threading.Lock()

    def update_estimate(self, last_epoch_time):
        with self.lock:
            self.estimated_remaining = last_epoch_time * (self.total_epochs - self.current_epoch) + time.time() - self.time

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
                    f"| Batch: {self.current_batch}/{self.total_batches}"
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

                progress_line = f"{epoch_progress} {batch_progress} | [{bar}] {round((time.time() - self.time))}s / {round(self.estimated_remaining) if self.estimated_remaining > 0 else "ETA"}s\r"
                sys.stdout.write(progress_line)
                sys.stdout.flush()
            time.sleep(0.1)  # Update interval