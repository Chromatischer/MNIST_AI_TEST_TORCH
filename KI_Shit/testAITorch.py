import time
import torch
import torch.nn as nn
import torch.optim as optim

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Erstellen der Trainingsdaten
inputMuster = torch.tensor([[1.0], [2.0], [4.0], [8.0], [10.0]], device=device)  # Shape: (5, 1)
outputMuster = torch.tensor([[3.0], [6.0], [12.0], [24.0], [30.0]], device=device)  # Shape: (5, 1)

# Aufbau des neuronalen Netzwerks
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Input size = 1, Output size = 1

    def forward(self, x):
        return self.linear(x)


# Model, Loss, and Optimizer
model = LinearModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Trainieren des neuronalen Netzwerks
start = time.time()
last_time = start
this_time = start

epochs = 50_000
for epoch in range(epochs):
    model.train()

    # Forward pass
    predictions = model(inputMuster)
    loss = criterion(predictions, outputMuster)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    last_time = this_time
    this_time = time.time()
    epoch_time = (this_time - last_time) * 1000

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Time: {epoch_time:.2f} ms")

end = time.time()
training_time = end - start
print(f"Training took: {training_time:.2f} seconds")

# Testen des neuronalen Netzwerks mit Testdaten
model.eval()
testMuster = torch.tensor([[22.0], [44.0]], device=device)  # Shape: (2, 1)
testPrediction = model(testMuster).detach().cpu().numpy()  # Detach and move to CPU for printing
print("Prediction for testMuster:", testPrediction)

# Model Weights and Bias
weights = model.linear.weight.detach().cpu().numpy()
bias = model.linear.bias.detach().cpu().numpy()
print("Model Weights (W):", weights)
print("Model Bias (b):", bias)