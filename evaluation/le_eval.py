from lighteval.models.abstract_model import LightevalModel
from lighteval.tasks.requests import Doc
import sys
import os
sys.path.insert(1, os.getcwd())
import torch
from typing import List, Tuple
import numpy as np

class CustomPyTorchModel(LightevalModel):
    def __init__(self, model, tokenizer, device='cuda', batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._batch_size = batch_size
        
        self.model.to(device)
        self.model.eval()
    
    @property
    def max_length(self):
        return 2048
    
    @property
    def tokenizer(self):
        """Return the tokenizer"""
        return self._tokenizer
    
    @property
    def add_special_tokens(self):
        """Whether to add special tokens"""
        return True
    
    def greedy_until(self, requests, disable_tqdm=False):
        """Generate text until stop sequences"""
        results = []
        
        for request in requests:
            context = request.context
            stop_sequences = request.stop_sequences or []
            
            # Tokenize
            input_ids = torch.tensor([self.tokenizer.encode(context)]).to(self.device)
            
            # Generate
            generated = input_ids[0].tolist()
            max_new_tokens = request.generation_size or 256
            
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    next_token = outputs[0, -1, :].argmax().item()
                
                generated.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(self.device)], dim=1)
                
                # Check for stop sequences
                text = self.tokenizer.decode(generated[len(input_ids[0])-len(generated):])
                if any(stop in text for stop in stop_sequences):
                    break
            
            generated_text = self.tokenizer.decode(generated[len(request.context):])
            results.append(generated_text)
        
        return results
    
    def loglikelihood(self, requests, disable_tqdm=False):
        """Calculate log likelihood for multiple choice questions"""
        results = []
        
        for request in requests:
            context = request.context
            continuation = request.choice
            
            # Tokenize context and full sequence
            context_ids = self.tokenizer.encode(context)
            full_ids = self.tokenizer.encode(context + continuation)
            continuation_ids = full_ids[len(context_ids):]
            
            # Get logits
            input_ids = torch.tensor([full_ids]).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            # Calculate log probabilities for continuation tokens
            log_probs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
            
            # Sum log probs for the continuation
            total_log_prob = 0.0
            for i, token_id in enumerate(continuation_ids):
                total_log_prob += log_probs[len(context_ids) + i - 1, token_id].item()
            
            # Check if this would be greedy choice
            is_greedy = True
            for i, token_id in enumerate(continuation_ids):
                greedy_token = outputs[0, len(context_ids) + i - 1].argmax().item()
                if greedy_token != token_id:
                    is_greedy = False
                    break
            
            results.append((total_log_prob, is_greedy))
        
        return results
    
    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        """Calculate rolling log likelihood"""
        results = []
        
        for request in requests:
            text = request.context
            token_ids = self.tokenizer.encode(text)
            
            if len(token_ids) < 2:
                results.append(0.0)
                continue
            
            input_ids = torch.tensor([token_ids]).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            log_probs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
            
            # Sum log probs for all tokens (predicting next token)
            total_log_prob = 0.0
            for i in range(len(token_ids) - 1):
                total_log_prob += log_probs[i, token_ids[i + 1]].item()
            
            results.append(total_log_prob)
        
        return results

# Use it with lighteval
from models.test_main_model_v1 import LLM_v1
from tokenizer.variants import BPE_tokenizer
import json

# Load config
with open('./training/sample.json') as f:
    config = json.load(f)

# Load your model
model = LLM_v1(config)
state_dict = torch.load('./models/main_model_v1', map_location='cpu')

# Clean compiled state dict
cleaned_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('_orig_mod.', '')
    cleaned_state_dict[new_key] = value

model.load_state_dict(cleaned_state_dict)

# Load tokenizer
tokenizer = BPE_tokenizer(bpe_type='cl100k_base')

# Wrap model
custom_model = CustomPyTorchModel(model, tokenizer, device='cuda', batch_size=8)