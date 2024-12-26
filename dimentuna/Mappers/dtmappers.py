import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class DTMapper(nn.Module, ABC):

    def __init__(self):
        super(DTMapper, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

class LinearMapper(DTMapper):
    def __init__(self, input_dim : int , output_dim : int, hidden_dim : list[int]| int = []):
        
        super(LinearMapper, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU(0.01))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        
        assert x.shape[2] == self.input_dim, f"Input dimension mismatch. Expected {self.input_dim} but got {x.shape[2]}"

        x = self.network(x)

        return x

