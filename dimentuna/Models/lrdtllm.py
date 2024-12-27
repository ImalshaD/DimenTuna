from .dtconfig import DTConfig
from .dtllm import DTHfLLM
from ..Utils import DTLayerWrapper

class LayerWrappebleDTHfLLM(DTHfLLM):
    def __init__(self, config: DTConfig):
        super().__init__(config)
        self.wrapper_mapping = dict()
    
    def replace_layer(self, layer_idx : int, layer_wrapper : DTLayerWrapper):
        original_layer = self.model.model.layers[layer_idx]
        layer_wrapper.set_layer(original_layer)
        self.model.model.layers[layer_idx] = layer_wrapper
        self.wrapper_mapping[layer_idx] = layer_wrapper
    
    def engage_layer_wrapper(self, layer_idx : int| None = None, status : bool = True):
        
        if layer_idx is None:
            return
        if layer_idx in self.wrapper_mapping:
            self.wrapper_mapping[layer_idx].engage(status)

    def engage_all_layer_wrappers(self, status : bool = True):
        for layer_idx in self.wrapper_mapping:
            self.wrapper_mapping[layer_idx].engage(status)
    
    def freeze_layer_wrapper(self, layer_idx : int, freeze_mapper : bool = True):
        if layer_idx in self.wrapper_mapping:
            self.wrapper_mapping[layer_idx].freeze(freeze_mapper)

    def freeze_all_wrappers(self, freeze_mapper : bool = True):
        for layer_idx in self.wrapper_mapping:
            self.freeze_layer_wrapper(layer_idx, freeze_mapper)
    
    def ready2train(self, layer_idx : int, engage_all : bool = False, engage_specific : list[int] = []):
        
        super().freeze()
        self.freeze_layer_wrapper(layer_idx, freeze_mapper=False)
        self.engage_all_layer_wrappers(status=False)
        
        if engage_all:
            self.engage_all_layer_wrappers()
        
        elif len(engage_specific) > 0:
            engage_specific.append(layer_idx)
            for layer_idx in engage_specific:
                self.engage_layer_wrapper(layer_idx)
        else:
            self.engage_layer_wrapper(layer_idx)
        
        if layer_idx in self.wrapper_mapping:
            return self.wrapper_mapping[layer_idx]
        else:
            raise ValueError(f"Layer {layer_idx} is not wrapped")
    
    def is_wrapper_engaged(self, layer_idx : int):
        if layer_idx not in self.wrapper_mapping:
            return False
        return self.wrapper_mapping[layer_idx].get_engage_status()
    
    def print_status(self):
        print("-LLM")
        
        for i, layer in enumerate(self.model.model.layers):    
            if i in self.wrapper_mapping:
                self.wrapper_mapping[i].print_stats(i)
            else:
                print(f"--Layer_{i}")
                print(f"---Layer_Frozen: {not any(p.requires_grad for p in layer.parameters())}")
                
    

        

    