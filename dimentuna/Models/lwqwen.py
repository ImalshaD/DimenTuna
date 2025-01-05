from .lrdtllm import LayerWrappebleDTHfLLM
from .dtconfig import DTConfig
import torch

class LayerWrappebleQwen(LayerWrappebleDTHfLLM):

    def __init__(self, max_generation_length: int = 512, 
                 do_sample: bool =  False, 
                 temperature: float = 0.7,
                 device: torch.device = None):
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        qwen_config = DTConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            cache_dir="./cache",          # Path to cache directory
            max_tokens=None,               # Maximum token limit
            temperature=temperature,              # Sampling temperature
            max_generation_length=max_generation_length,    # Maximum length for generated text
            do_sample=do_sample,               # Enable sampling
            truncation=True,              # Enable truncation
            padding=True,                 # Enable padding
            device=device,  # Device configuration
            use_best_config=True,        # Use best configuration (custom logic)
            padding_side='left',         # Padding side
            output_hidden_states=False,   # Output hidden states
            output_attentions=False,      # Output attention values
            return_dict=True,             # Return output as dictionary
        )
        super().__init__(qwen_config)

        self.best_config = {
            "max_length": max_generation_length
        }
    
    def applyTemplate(self, texts, **kwargs):
        
        system_prompt = kwargs.get("system_prompt", None)
        
        if system_prompt is None:
            raise ValueError("system_prompt is required for Qwen")

        batch_messages = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{prompt}"}
            ]
            for prompt in texts
        ]

        batch_texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in batch_messages
        ]

        return batch_texts