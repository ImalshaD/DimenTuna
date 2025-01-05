from .dtconfig import DTConfig

import torch
from transformers import AutoTokenizer, AutoModel

from typing import Optional, Callable

class DTHfEncoder:
    
    def __init__(self, config: DTConfig):
        self.tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir, padding_side=config.padding_side)
        self.model = AutoModel.from_pretrained(config.model_name, cache_dir=config.cache_dir)
        self.device = config.device

        self.max_tokens = config.max_tokens
        self.padding = config.padding
        self.truncation = config.truncation
        self.template = config.template

        self.embedding_size = self.model.config.hidden_size

        self.to()
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def to(self, device=None):
        if device is None:
            device = self.device
        else:
            self.device = device
        if device is not None:
            self.model.to(device)

    def enableDP(self, gpu_ids=None):
        if self.device == torch.device('cpu'):
            raise ValueError("Device is CPU. Can't use DataParallel")
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu_ids)
    
    def get_embeddings_size(self):
        return self.embedding_size
    
    def getEmbedding_shape(self):
        return (self.max_tokens,self.embedding_size)
    
    def applyTemplate(self,texts, **kwargs):
        if self.template is not None:
            return [self.template.format(text=text) for text in texts]
        return texts
    
    def tokenize(self, texts, **kwargs):
        texts = self.applyTemplate(texts, **kwargs)
        inputs = self.tokenizer(texts, max_length=self.max_tokens, padding=self.padding, truncation=self.truncation, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs
    
    def encode(self, texts, pooling_strategy=None, custom_function: Optional[Callable]= None):
        inputs = self.tokenize(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        
        if pooling_strategy is None and custom_function is None:
            return embeddings
        
        if pooling_strategy == 'mean':
            embeddings = embeddings.mean(dim=1)
        elif pooling_strategy == 'max':
            embeddings = embeddings.max(dim=1)
        elif pooling_strategy == 'cls':
            embeddings = embeddings[:,0,:]
        elif pooling_strategy == "pooler":
            embeddings = outputs.pooler_output
        elif custom_function is not None:
            embeddings = custom_function(embeddings)
        else:
            raise ValueError(f"Pooling strategy {pooling_strategy} not supported")
        return embeddings
    

    
