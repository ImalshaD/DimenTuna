from Models import DTHfEncoder, LayerWrappebleDTHfLLM
from ..Projectors import DTProjector
from .dttrainstrat import DTTrainStratergy

class DTTrainer:
    
    def __init__(self, encoder : DTHfEncoder, llmodel : LayerWrappebleDTHfLLM, projector : DTProjector,
                 target_layers : list[int]|int,
                 train_strat,
                 train_loader, val_loader, test_loader ,lr, weight_decay, device,
                 **kwargs
                 ):
        
        self.encoder : DTHfEncoder = encoder
        self.llmodel : LayerWrappebleDTHfLLM = llmodel
        self.projector : DTProjector = projector
        self.train_strat : DTTrainStratergy = train_strat(llmodel, encoder, projector,
                                                train_loader, val_loader, test_loader, lr, weight_decay, device, **kwargs)
        
        if isinstance(target_layers, int):
            target_layers = [target_layers]

        self.target_layers = target_layers
    
    def train_layer(self, layer_idx : int, 
                    engage_all : bool = False, engage_specific : list[int] = []):
        
        self.train_strat.train()

    
    
    
