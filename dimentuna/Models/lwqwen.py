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

        self.eos_token_id = 151645
        self.sos_token_id = 151644

    def applyTemplate(self, texts, **kwargs):
        
        system_prompt = kwargs.get("system_prompt", None)
        
        if system_prompt is None:
            system_prompt = self.system_prompt

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
    
    
    def generate_user_mask(self, tokenized_texts, **kwargs):
        
        input_ids = tokenized_texts["input_ids"]
        results = []

        for vec in input_ids:
            
            vec_results = []
            
            eos_count=0
            sos_count=0
            passes=0
            
            for token in vec:

                if token == self.eos_token_id:
                    eos_count+=1
                    passes=0
                elif token == self.sos_token_id:
                    sos_count+=1
                else:
                    passes += 1

                if sos_count==2 and passes>3:
                    vec_results.append(1)
                else:
                    vec_results.append(0)
                    
            results.append(vec_results)
        return torch.tensor(results)
    
    def tokenize(self, texts, **kwargs):
        tokenized_texts = super().tokenize(texts, **kwargs)
        user_mask = self.generate_user_mask(tokenized_texts)
        user_mask.to(tokenized_texts["input_ids"].device)
        tokenized_texts["attention_mask"] += user_mask
        return tokenized_texts

    def filterByUserMaskandDecode(self, tokenized_texts):
        input_ids = tokenized_texts["input_ids"]
        user_mask = (tokenized_texts["attention_mask"]==2)
        results = []
        for vec, mask in zip(input_ids, user_mask):
            results.append([token for token, mask_val in zip(vec, mask) if mask_val==1])
        return self.tokenizer.batch_decode(results, skip_special_tokens=True)