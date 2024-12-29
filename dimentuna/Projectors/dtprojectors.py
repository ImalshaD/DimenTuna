import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class DTProjector(nn.Module, ABC):

    def __init__(self):
        super(DTProjector, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass


class FeedForwardProjector(DTProjector):
    def __init__(self, input_dim : int , output_dim : int , hidden_dim : list[int]| int = [], pooler_type='mean', pooler_function=None):
        super(FeedForwardProjector, self).__init__()
        
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        elif hidden_dim is None:
            hidden_dim = []
        

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.pooler_type = pooler_type
        self.pooler_function = pooler_function
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU(0.01))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        
    
    def forward(self, x):
        
        if x.dim() == 3:
            
            if self.pooler_type == 'mean':
                x = x.mean(dim=1)
            elif self.pooler_type == 'max':
                x = x.max(dim=1)
            elif self.pooler_type == "pooler":
                x = x[:,0,:]
            elif self.pooler_function is not None:
                x = self.pooler_function(x)
            else:
                raise ValueError(f"Invalid Pooler Type {self.pooler_type}")
        
        assert x.shape[1] == self.input_dim, f"Input dimension mismatch. Expected {self.input_dim} but got {x.shape[1]}"
        
        x = self.network(x)

        return x

class LSTMProjector(DTProjector):
    def __init__(self, input_dim ,output_dim, hidden_dim, num_layers=1, bidirectional=False):
        super(LSTMProjector, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        
        x, (hidden, cell) = self.lstm(x)
        output = self.fc(hidden)

        return output

class TransformerProjector(DTProjector):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=512, dropout=0.1):
        super(TransformerProjector, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.input_fc = nn.Linear(input_dim, hidden_dim)  # Project input to hidden_dim
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim, output_dim)  # Project to output_dim
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            Tensor of shape [batch_size, seq_len, output_dim]
        """
        assert x.shape[2] == self.input_dim, f"Input dimension mismatch. Expected {self.input_dim} but got {x.shape[2]}"
        
        # Project input to hidden dimension
        x = self.input_fc(x)
        
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Project to output dimension
        x = self.fc(x)
        x = self.tanh(x)
        
        return x
