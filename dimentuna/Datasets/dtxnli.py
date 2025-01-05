from .dtdataset import DTDataset
from datasets import load_dataset
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class DTXNLI(DTDataset):
    
    def __init__(self, cache_dir: str = None, batch_size: int = 1):
        system_prompt = """The task is to solve Natural Language Inference (NLI) problems. 
                            NLI is the task of determining the inference relation between two (short, ordered) texts: entailment, contradiction, or neutral. 
                            Answer as concisely as possible in the same format below: 
                            format -> [Answer: Entailment | Contradiction | Neutral]"""
        super().__init__(cache_dir, system_prompt,batch_size)

        self.class_labels = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }
        self.prompt_query = "Premise: [{premise}] \n Hypothesis: [{hypothesis}] Entailment, Contradiction, or Neutral?"
    
    def get_dataset(self, language: str):
        dataset = load_dataset("facebook/xnli", language, cache_dir=self.cache_dir)
        return dataset
    
    def get_dataset_split(self, language: str, split: str):
        dataset = self.get_dataset(language)
        return dataset[split]
    
    def get_columns(self, language: str):
        dataset = self.get_dataset(language)
        return dataset.column_names
    
    def calculate_metrics(self, generation, target, **kwargs):
        
        accuracy = accuracy_score(target, generation)
        return {"accuracy": accuracy}
    
    def _get_lable(self, generation):
        
        match = re.search(r'\b(Entailment|Contradiction|Neutral)\b', generation, re.IGNORECASE)
        
        if match:
            answer = match.group(1).lower()
            return self.class_labels[answer]
        else:
            return 3
    
    def _evaluate(self, model, language, batch_size = None, **kwargs):

        results = dict()
        
        if batch_size is None:
            batch_size = self.batch_size

        features = ["premise", "hypothesis"]
        target = "label"
        
        split = "test"

        dataloader = self.get_dataloader(language, split, features, target, template=self.prompt_query, batch_size=batch_size)
        
        for batch in  tqdm(dataloader, desc=f"Evaluating {language}"):
            queries, labels = batch
            generation = model.generate(queries, system_prompt=self.system_prompt)
            generated_labels = [self._get_lable(g) for g in generation]
            
            metrics = self.calculate_metrics(generated_labels, labels)
            
            results = self.update_results(results, metrics)
        
        return self.divide_results(results, len(dataloader))
    
    def print_sample(self, model, language, batch_size = None, **kwargs):

        if batch_size is None:
            batch_size = self.batch_size

        features = ["premise", "hypothesis"]
        target = "label"
        
        split = "test"

        dataloader = self.get_dataloader(language, split, features, target, template=self.prompt_query,batch_size=batch_size)
        
        for batch  in dataloader:
            
            queries, labels = batch
            generation = model.generate(queries, system_prompt=self.system_prompt)
            generated_labels = [self._get_lable(g) for g in generation]
            
            for q, l, g, gl in zip(queries, labels, generation ,generated_labels):
                print(f"""query: {q} \n target: {l} \n generated: {g} \n generated label: {gl}""")
            break