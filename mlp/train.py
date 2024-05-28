import torch.nn.functional as F

import numpy as np
import torch
import matplotlib.pyplot as plt

def select_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        
    else:
        device = torch.device("cpu")

    print(f"using device: {device}")
    return device

def train_model(model, optimizer, train_loader, n_epochs, device):
    """
    Args:
      model: The model to be trained.
      optimizer: The optimizer object.
      train_loader: The DataLoader object that contains the training data.
      n_epochs: The number of epochs to train the model.
      device: The device to train the model on.
    """
    # Set the model to training mode
    model.to(device).train() # model should be in device already.

    epoch_avg_loss = [] # store average loss per epoch
    
    for epoch in range(n_epochs):
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad() # Zero the gradients
            
            y_pred = model(x.to(device)) # Forward pass
            loss = F.mse_loss(y_pred.to(device), y.to(device)) # Calculate the loss
            
            # Backward pass and take a step with the optimizer
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() # accumulate loss for average loss per epoch
            
        avg_loss = total_loss / len(train_loader)
        epoch_avg_loss.append(avg_loss) # store average loss per epoch
    
    plt.plot(epoch_avg_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss')
    plt.show()
    
    save = input("Save the model? (y/n): ")
    if save.lower() == 'y':
        torch.save(model.state_dict(), 'mlp.pth')
        print("Model saved.")
    else:
        print("Model not saved.")
        
    return model, epoch_avg_loss

def generate_data():
    np.random.seed(2)
    x = np.linspace(-1, 1, 100).reshape(-1, 1)
    
    # some random polynomial function for the pattern
    y = 2*x**3 - x**2 + 0.5 * x
    
    # Add some noise
    y += 0.1 * np.random.randn(*x.shape)
    
    # Convert to PyTorch tensors
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    
    return x, y


