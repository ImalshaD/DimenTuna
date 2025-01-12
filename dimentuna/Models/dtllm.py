from .dtconfig import DTConfig
from .dtencoder import DTHfEncoder

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2ForQuestionAnswering

from typing import Optional, Callable

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

        self.system_prompt = None

        self.to()
    
    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt
    
    def change_best_config(self, max_generation_length: int = None, do_sample: bool = None, temperature: float = None, **kwargs):
        
        best_config = {
            "max_length": max_generation_length if max_generation_length is not None else self.best_config["max_length"],
            "do_sample": do_sample if do_sample is not None else self.best_config["do_sample"],
            "temperature": temperature if temperature is not None else self.best_config["temperature"]
        }
        best_config.update(kwargs)
        self.best_config = best_config
    
    def print_best_config(self):
        print(self.best_config)
        
        
    def get_Layer_output(self, texts, layer_idx, pooling_strategy=None, custom_function: Optional[Callable] = None):
        
        layer_idx +=1 # TODO: Check if this is correct
        inputs = self.tokenize(texts)
        
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
        
        del inputs, outputs
        torch.cuda.empty_cache()
        
        return layer_output
    
    def generate(self, texts,**kwargs):
        inputs = self.tokenize(texts, **kwargs)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **self.best_config)
            outputs = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)]
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts