from .dtdataset import DTDataset
from datasets import load_dataset
import re
from tqdm import tqdm

class DTMsgm(DTDataset):
    
    def __init__(self, cache_dir: str = None, batch_size: int = 1):
        system_prompt = "Please reason step by step, and put your final answer number in a newline within square brackets."
        super().__init__(cache_dir, system_prompt,batch_size)
    
    def get_dataset(self, language: str):
        dataset = load_dataset("juletxara/mgsm", language, cache_dir=self.cache_dir)
        return dataset
    
    def get_dataset_split(self, language: str, split: str):
        dataset = self.get_dataset(language)
        return dataset[split]
    
    def get_columns(self, language: str):
        dataset = self.get_dataset(language)
        return dataset.column_names
    
    def calculate_metrics(self, generation, target, **kwargs):
        
        correct = 0
        pattern = r"\[(.*?)\]"
        
        for response, correct_ans in zip(generation, target):
            matches = re.findall(pattern, response)
            if len(matches) > 0 and matches[-1] == str(correct_ans):
                correct += 1
        
        return {"accuracy": correct/len(generation)}
    
    def _evaluate(self, model, language, batch_size = None, **kwargs):

        results = dict()
        
        if batch_size is None:
            batch_size = self.batch_size

        features = ["question"]
        target = "answer_number"
        
        split = "test"

        dataloader = self.get_dataloader(language, split, features, target, batch_size=batch_size)
        
        for batch in  tqdm(dataloader, desc=f"Evaluating {language}"):
            queries, targets = batch
            generation = model.generate(queries, system_prompt=self.system_prompt)
            metrics = self.calculate_metrics(generation, targets)
            results = self.update_results(results, metrics)
        
        return self.divide_results(results, len(dataloader))
    
    def print_sample(self, model, language, batch_size = None, **kwargs):
        
        if batch_size is None:
            batch_size = self.batch_size

        features = ["question"]
        target = "answer_number"

        split = "test"

        dataloader = self.get_dataloader(language, split, features, target, batch_size=batch_size)

        for batch in dataloader:
            
            queries, targets = batch
            generation = model.generate(queries, system_prompt=self.system_prompt)
            
            for query, target, gen in zip(queries, targets, generation):
                print( f"""
                Query: {query}
                Correct Answer: {target}
                Generated Answer: {gen}
                """)
            break
    
    
    
        
        