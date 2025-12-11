from lighteval.models.abstract_model import LightevalModel
import sys
import os
sys.path.insert(1, os.getcwd())
from models.test_main_model_v1 import LLM_v1
from models.instruct_model import Instruct_Model_v1
from tokenizer.variants import BPE_tokenizer
import os
from lighteval.data import GenerativeTaskDataset
from typing import List
from lighteval.tasks.requests import (
    Doc,
)
from lighteval.models.model_output import (
    ModelResponse,
)
from tqdm import tqdm
import torch


# class MyCustomModel(LightevalModel):
#     def __init__(self, config):
#         super().__init__()
#         # Initialize your model here...
#         print(os.getcwd())
#         self._tokenizer = BPE_tokenizer()
#         self.model = LLM_v1(config)
#         self.model = torch.compile(self.model, mode='default')
#         self.model.load_state_dict(torch.load('./models/main_model_v1'))

#     def greedy_until(self, requests: List[Doc], max_tokens:int=2048) -> List[ModelResponse]:
#         # Implement generation logic
#         for request in requests:
#             request.tokenized_context = self.tokenizer.encode(request.context)
        
#         dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
#         for split in tqdm(
#             dataset.splits_iterator(),
#             desc='Splits',
#             position=0,
#             disable=False
#         ):
#             for r in tqdm(split, desc="Batch", position=1, disable=False):
#                 # Extract source and target languages from task name
#                 # Format is like "community|sdst-text_level:de-fr|0"
#                 respose = 

#                 cur_response = ModelResponse(
#                     result=result,
#                     logits=None,
#                     generated_tokens=[],
#                     input_tokens=[],
#                 )
#                 results.append(cur_response)

#         results = []

#         pass

#     @property
#     def tokenizer(self):
#         return self._tokenizer

#     def tok_encode(self, text: str):
#         return text

#     @property
#     def add_special_tokens(self) -> bool:
#         return True

#     @property
#     def max_length(self) -> int:
#         """Return the maximum sequence length of the model."""
#         return 2048

#     def loglikelihood(self, requests: list[Doc]) -> list[ModelResponse]:
#         """Tokenize the context and continuation and compute the log likelihood of those
#         tokenized sequences.
#         """
#         raise NotImplementedError

#     def loglikelihood_rolling(
#         self,
#         requests: list[Doc],
#     ) -> list[ModelResponse]:
#         """This function is used to compute the log likelihood of the context for perplexity metrics."""
#         raise NotImplementedError

#     def loglikelihood_single_token(self, requests):
#         # Implement single token loglikelihood computation
#         raise NotImplementedError
    

import torch
from datasets import load_dataset
from tqdm import tqdm
import json

# Load model

print("Loading model...")
with open('./training/sample.json') as f:
    model = LLM_v1(json.load(f))
model = torch.compile(model, mode='default')
model = Instruct_Model_v1(model, {
        'Wqkv': '',
        'out_proj': '',
        'router': '',
        'w1': '',
        'w2': '',
        'w3': ''
    }, {
        'rank': 32,
        'alpha': 32,
        'dropout':0.0
    })
model.load_state_dict(torch.load('./models/instruct_model_v3'))
tokenizer = BPE_tokenizer()
model = model.cuda().eval()

# Get answer token IDs
answer_tokens = []
for letter in [" A", " B", " C", " D"]:
    tokens = tokenizer.encode(letter)
    answer_tokens.append(tokens[-1] if tokens else tokenizer.encode(letter.strip())[0])
print(answer_tokens)

print(f"Answer tokens: {answer_tokens}")

# All MMLU subjects
SUBJECTS = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
            'clinical_knowledge', 'college_biology', 'college_chemistry',
            'college_computer_science', 'college_mathematics', 'college_medicine',
            'college_physics', 'computer_security', 'conceptual_physics',
            'econometrics', 'electrical_engineering', 'elementary_mathematics',
            'formal_logic', 'global_facts', 'high_school_biology',
            'high_school_chemistry', 'high_school_computer_science',
            'high_school_european_history', 'high_school_geography',
            'high_school_government_and_politics', 'high_school_macroeconomics',
            'high_school_mathematics', 'high_school_microeconomics',
            'high_school_physics', 'high_school_psychology', 'high_school_statistics',
            'high_school_us_history', 'high_school_world_history', 'human_aging',
            'human_sexuality', 'international_law', 'jurisprudence',
            'logical_fallacies', 'machine_learning', 'management', 'marketing',
            'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
            'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
            'professional_law', 'professional_medicine', 'professional_psychology',
            'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
            'virology', 'world_religions']

# Evaluation
total_correct = 0
total_questions = 0
results_by_subject = {}

print("\nEvaluating MMLU (0-shot)...")

for subject in tqdm(SUBJECTS):
    
        dataset = load_dataset("cais/mmlu", subject)
        
        # Get few-shot examples
        fewshot = []
        if 'dev' in dataset:
            fewshot = list(dataset['dev'])[:5]
        elif 'validation' in dataset:
            fewshot = list(dataset['validation'])[:5]
        correct = 0
        total = 0
        
        for example in dataset['test']:
            # Build prompt with few-shot examples
            prompt = ""
            
            for ex in fewshot:
                prompt += f"Question: {ex['question']}\n"
                for i, choice in enumerate(ex['choices']):
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += f"Answer: {chr(65+ex['answer'])}\n\n"
            
            # Add test question
            prompt += f"Question: {example['question']}\n"
            for i, choice in enumerate(example['choices']):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "Answer:"
            
            # Get prediction
            inputs = torch.tensor(tokenizer.encode(prompt), device='cuda')
            inputs = inputs.unsqueeze(0)
            with torch.no_grad():
                with torch.autocast(device_type='cuda:0', dtype=torch.bfloat16):
                    logits = model(inputs)
                last_token_logits = logits[0, -1]
            
            # Compare A/B/C/D logits
            ans_logits = last_token_logits[answer_tokens]
            predicted = ans_logits.argmax().item()

            if predicted == example['answer']:
                correct += 1
            total += 1
        
        accuracy = correct / total
        results_by_subject[subject] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
        
        total_correct += correct
        total_questions += total
        
        tqdm.write(f"{subject}: {accuracy:.2%} ({correct}/{total})")
        
    # except Exception as e:
    #     tqdm.write(f"Error on {subject}: {e}")

# Final results
overall_accuracy = total_correct / total_questions

print("\n" + "="*70)
print(f"MMLU ACCURACY: {overall_accuracy:.2%} ({total_correct}/{total_questions})")
print("="*70)

# Save results
with open('mmlu_baseline_results_0shot.json', 'w') as f:
    json.dump({
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_questions': total_questions,
        'num_fewshot': 0,
        'by_subject': results_by_subject
    }, f, indent=2)

print("\nResults saved to mmlu_baseline_results.json")