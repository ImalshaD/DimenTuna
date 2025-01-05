from abc import ABC, abstractmethod
from ..Models import DTHfLLM
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DTCustomDataset(Dataset):

    def __init__(self, data, features : list[str], target : str = None ,template: str = None):
        
        self.data = data
        self.features = features
        self.target = target
        self.template = template
        self.target_appended_features = features + [target] if target is not None else features
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        if self.template is not None:
            return self.template.format(**{col: self.data[col][idx] for col in self.features}), self.data[self.target][idx]
        else:
            return [self.data[col][idx] if self.data[col][idx] != None else "Empty" for col in self.target_appended_features]
    

class DTDataset(ABC):
    
    def __init__(self, cache_dir: str = None , system_prompt: str = None, batch_size: int = 1):
        self.cache_dir = cache_dir
        self.system_prompt = system_prompt
        self.batch_size = batch_size
    
    @abstractmethod
    def get_dataset(self, language: str):
        pass

    @abstractmethod
    def get_dataset_split(self, language: str, split: str):
        pass

    @abstractmethod
    def get_columns(self, language: str):
        pass
    
    @abstractmethod
    def calculate_metrics(self, genetation, target, **kwargs)-> dict:
        pass

    @abstractmethod
    def _evaluate(self, model: DTHfLLM, language: str, batch_size: int = None, **kwargs):
        pass
    
    @abstractmethod
    def print_sample(self, model: DTHfLLM, language: str, batch_size: int = None, **kwargs):
        pass
    def get_dataloader(self, language: str, split: str, features: list[str], target : str = None, template : str = None,batch_size: int = None):
        
        if batch_size is None:
            batch_size = self.batch_size
        
        dataset = self.get_dataset_split(language, split)
        cus_dataset = DTCustomDataset(dataset, features, target, template)

        return DataLoader(cus_dataset, batch_size=batch_size, shuffle=True)
    
    def get_dataloaders(self, language: str, splits: list[str], columns: list[str], batch_size: int = None):        
        
        if len(splits) == 0:
            raise ValueError("No splits provided")
        elif len(splits) == 1:
            return self.get_dataloader(language, splits[0], columns, batch_size)
        else:
            return [self.get_dataloader(language, split, columns, batch_size) for split in splits]
    
    def update_results(self, results: dict, batch_results: dict):
        for key in batch_results:
            if key in results:
                results[key] += batch_results[key]
            else:
                results[key] = batch_results[key]
        return results

    def divide_results(self, results: dict, n: int):
        for key, value in results.items():
            results[key] = value/n
        return results
    
    def evaluate(self, model: DTHfLLM, language: list[str], batch_size: int = None, **kwargs):
        
        results = dict()
        
        for lang in language:
            results[lang] = self._evaluate(model, lang, batch_size, **kwargs)
        
        return results
            
        
        

        
        