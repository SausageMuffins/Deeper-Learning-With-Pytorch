import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, n_inputs=1):
        super(MLP, self).__init__() # inherit from superclass nn.Module
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 11)
        self.fc3 = nn.Linear(11, 1)
        
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()

    def forward(self, x):
        """
        Args:
          x of shape (n_samples, n_inputs): Model inputs.
        
        Returns:
          y of shape (n_samples, 1): Model outputs.
        """
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh2(x)
        y = self.fc3(x)
        return y