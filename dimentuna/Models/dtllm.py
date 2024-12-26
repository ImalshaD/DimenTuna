from .dtconfig import DTConfig
from .dtencoder import DTHfEncoder

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class DTHfLLM(DTHfEncoder):

    def __init__(self, config: DTConfig):
        self.tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir, padding_side=config.padding_side)
        self.model : AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(config.model_name, cache_dir=config.cache_dir)
        
        self.device = config.device

        self.max_tokens = config.max_tokens
        self.padding = config.padding
        self.truncation = config.truncation
        self.template = config.template

        self.embedding_size = self.model.config.hidden_size
        self.use_best_config = config.use_best_config
        
        self.best_config = None if not self.use_best_config else {
            "max_length": config.max_generation_length,
            "do_sample": config.do_sample,
            "temperature": config.temperature
        }
        
    def get_Layer_output(self, texts, layer_idx, pooling_strategy=None, custom_function: None | callable = None):
        inputs = self.tokenize(texts)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            layer_output = outputs.hidden_states[layer_idx]
        
        if pooling_strategy is None and custom_function is None:
            return layer_output
        
        if pooling_strategy == 'mean':
            layer_output = layer_output.mean(dim=1)
        elif pooling_strategy == 'max':
            layer_output = layer_output.max(dim=1)
        elif custom_function is not None:
            layer_output = custom_function(layer_output)
        else:
            raise ValueError("Invalid pooling strategy or custom function")
        
        return layer_output
    
    def generate(self, texts):
        inputs = self.tokenize(texts)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.best_config)
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts