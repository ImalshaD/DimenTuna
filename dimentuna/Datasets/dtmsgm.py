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
    
    def extract_answer(self, response):
        pattern = r"\[(.*?)\]"
        matches = re.findall(pattern, response)
        return matches[-1] if len(matches) > 0 else "Empty Answer"

    def calculate_metrics(self, generation, target, **kwargs):
        logging = kwargs.get("logging", False)
        correct = 0
        for response, correct_ans in zip(generation, target):
            answer = self.extract_answer(response)
            if logging:
                print(f"answer: {answer}, correct: {correct_ans} correctness: {answer.strip() == str(correct_ans)}")
            if answer.strip() == str(correct_ans):
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
            metrics = self.calculate_metrics(generation, targets, **kwargs)
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
            extracted_answers = [self.extract_answer(response) for response in generation]
            
            for query, target, gen, ex_ans in zip(queries, targets, generation, extracted_answers):
                print( f"""Query: {query} \n Correct Answer: {target} \n Generated Answer: {gen} \n Extracted Answer: {ex_ans} \n""")
            break
    
    
    
        
        