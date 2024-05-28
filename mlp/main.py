import torch.nn
from torch.utils.data import DataLoader, TensorDataset

# import other modules
from model import MLP # the defined model
from train import train_model, select_device, generate_data # method to train the model
import matplotlib.pyplot as plt

# Select the device - cpu, gpu or metal to train the model
device = select_device()

model = MLP() # instantiate the model and move it to the device

optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # instantiate the optimizer, we fixed the lr to 0.001 but can be changed

x, targets = generate_data() # generate the data for training

# Creating a train loader to train my model - this is to load the data in batches + shuffle
dataset = TensorDataset(x, targets)
train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Train the model + save if happy with the loss
n_epochs = 300 # FEEL FREE TO CHANGE THIS
model, epoch_avg_loss = train_model(model, optimizer, train_loader, n_epochs, device)

# Evaluate the model
with torch.no_grad():
    y_pred = model(x.to(device)).cpu()

# Plot the original data and the model's predictions
fig, ax = plt.subplots(1)
ax.plot(x.numpy(), targets.numpy(), '.', label='Original Data')
ax.plot(x.numpy(), y_pred.numpy(), 'r-', label='MLP Predictions')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('MLP Predictions vs Original Data')
ax.legend()
ax.grid(True)
plt.show()
