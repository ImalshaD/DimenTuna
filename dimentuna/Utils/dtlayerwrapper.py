import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from ..Mappers import DTMapper

class DTLayerWrapper(ABC, nn.Module):
    
    def __init__(self, mapper : DTMapper, layer : nn.Module|None = None):
        super().__init__()
        self.layer = layer
        self.mapper : DTMapper = mapper
        self.engage_status  = True

    def print_stats(self, layer_idx : int):
        layer_forzen = not any(p.requires_grad for p in self.layer.parameters())
        mapper_forzen = not any(p.requires_grad for p in self.mapper.parameters())
        print(f"--LayerWrapper_{layer_idx}")
        print(f"---Layer_Frozen: {layer_forzen}")
        print(f"---Mapper_Frozen: {mapper_forzen}")
        print(f"---Engaged: {self.engage_status}")
    
    def __repr__(self):
        return super().__repr__()+ f"Engaged: {self.engage_status}"
    
    def set_layer(self, layer : nn.Module):
        self.layer = layer
    
    def engage(self, status : bool = True):
        self.engage_status = status

    def engage_status(self):
        return self.engage_status
    
    @abstractmethod
    def forward_pass(self, hidden_states, attention_mask=None, **kwargs):
        pass

    @abstractmethod
    def freeze(self):
        pass

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        if self.engage_status:
            return self.forward_pass(hidden_states, attention_mask=attention_mask, **kwargs)
        else:
            return self.layer(hidden_states, attention_mask=attention_mask, **kwargs)
        

class LinearWrapper(DTLayerWrapper):
    def __init__(self, mapper : nn.Module, layer : nn.Module|None = None):
        super().__init__(mapper, layer)

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.tanh = nn.Tanh()

    def forward_pass(self, hidden_states, attention_mask=None, **kwargs):
        
        layer_output = self.layer(hidden_states, attention_mask=attention_mask, **kwargs)
        
        if isinstance(layer_output, tuple):
            modified_hidden_states = layer_output[0] + self.mapper(layer_output[0])
            return (modified_hidden_states,) + layer_output[1:]
        else:
            return layer_output + self.mapper(layer_output)
    
    def print_stats(self, layer_idx):
        super().print_stats(layer_idx)
        alpha_forzen = not self.alpha.requires_grad
        print(f"---Alpha_Frozen: {alpha_forzen} Value: {self.alpha.item()} Not Used")
    
    def view_alpha(self):
        return self.alpha.item()
    
    def freeze(self, freeze_mapper : bool= True):
        self.alpha.requires_grad = not(freeze_mapper)
        for param in self.mapper.parameters():
            param.requires_grad = not(freeze_mapper)