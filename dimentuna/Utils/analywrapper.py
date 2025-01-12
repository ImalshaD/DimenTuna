from .dtlayerwrapper import DTLayerWrapper
import torch
import torch.nn as nn

class NoiseWrapper(DTLayerWrapper):

    def __init__(self,layer : nn.Module|None = None , bound : float = 1.0):
        super().__init__(None, layer)
        self.noise = bound

    def change_bound(self, bound : float):
        self.noise = bound

    def forward_pass(self, hidden_states, attention_mask=None, **kwargs):
        
        layer_output = self.layer(hidden_states, attention_mask=attention_mask, **kwargs)
        
        if isinstance(layer_output, tuple):
            modified_hidden_states = layer_output[0] + self.getRandomNoise(layer_output[0].shape)
            return (modified_hidden_states,) + layer_output[1:]
        else:
            return layer_output + self.getRandomNoise(layer_output.shape)
    
    def print_stats(self, layer_idx):
        super().print_stats(layer_idx)
        print(f"---Noise_Bound: {self.noise}")
    
    def getRandomNoise(self, shape):
        return (torch.rand(shape) * 2 -1) * self.noise
    
    def freeze(self, freeze_mapper : bool= True):
        if self.mapper is not None:
            for param in self.mapper.parameters():
                param.requires_grad = not(freeze_mapper)

class Observewrapper(DTLayerWrapper):

    def __init__(self,layer = None):
        mapper = None
        super().__init__(mapper, layer)
    
    def forward_pass(self, hidden_states, attention_mask=None, **kwargs):
        layer_output = self.layer(hidden_states, attention_mask=attention_mask, **kwargs)
        return layer_output

    def freeze(self):
        pass