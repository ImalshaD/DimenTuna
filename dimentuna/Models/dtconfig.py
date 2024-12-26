import os
import torch

class DTConfig:

    def __init__(self, model_name: str, cache_dir: str, max_tokens : int, 
                 temperature : float = 0.7, max_generation_length : int= 512,
                do_sample: bool=True, truncation: bool=True, padding : str|bool= True,
                device : None |torch.device =None, use_best_config : bool=False, padding_side : str='right',
                output_hidden_states : bool=False, output_attentions : bool=False, return_dict : bool=False,
                template : str=None
                ):
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_generation_length = max_generation_length
        self.device : torch.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding
        self.truncation = truncation
        self.temperature = temperature
        self.do_sample = do_sample
        self.cache_dir = cache_dir
        self.use_best_config = use_best_config
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.return_dict = return_dict
        self.padding_side = padding_side
        self.template = template

    def create_directories(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)