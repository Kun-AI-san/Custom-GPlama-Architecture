import torch
import torch.nn as nn
import math
from typing import Dict
from models.test_main_model_v1 import LLM_v1
import json
import os


class LoRAAdaptor(nn.Module):
    def __init__(self, d_in:int, d_out:int, rank:int=16, alpha:float=16.0, dropout:float=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha/rank

        self.matA = nn.Linear(d_in, rank, bias=False)
        self.matB = nn.Linear(rank, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.matA.weight)
        nn.init.zeros_(self.matB.weight)

    def forward(self, x):
        return self.matB(self.dropout(self.matA(x))) * self.scale
    

class LoRAWrapper(nn.Module):
    def __init__(self, orig_layer: nn.Linear, lora_location:str='', layer_number:int=-1, rank:int=16, alpha:float=16.0, dropout:float=0.0):
        super().__init__()
        self.original_layer = orig_layer
        self.lora_adaptor = LoRAAdaptor(orig_layer.in_features, orig_layer.out_features, rank, alpha, dropout)
        self.lora_location = lora_location
        if lora_location != '':
            if os.path.exists(self.lora_location):
                with open(self.lora_location) as j:
                    file = torch.load(j)
                    self.lora_adaptor.load_state_dict(file[layer_number])
        self.layer_number = layer_number
        if self.layer_number == -1:
            raise 'Layer number has not been initialized correctly, please check your initializations'
    def forward(self, x):
        return self.original_layer(x) + self.lora_adaptor(x)
    
class Instruct_Model_v1(nn.Module):
    def __init__(self, model:LLM_v1, lora_location:Dict, lora_config:Dict):
        super().__init__()
        self.foundation_model = model
        self.rank = lora_config['rank']
        self.alpha = lora_config['alpha']
        self.lora_location = lora_location
        self.lora_dropout = lora_config['dropout']

        for parameter in self.foundation_model.parameters():
            parameter.requires_grad = False
        
        self.foundation_model.out_head.requires_grad_ = True

        for i, block in enumerate(self.foundation_model.trf_blocks):
            block.multihead_attention.Wqkv = LoRAWrapper(block.multihead_attention.Wqkv, self.lora_location['Wqkv'], i, self.rank, self.alpha, self.lora_dropout)
            block.multihead_attention.out_proj = LoRAWrapper(block.multihead_attention.out_proj, self.lora_location['out_proj'], i, self.rank, self.alpha, self.lora_dropout)
            block.router.w1 = LoRAWrapper(block.router.w1, self.lora_location['w1'], i, self.rank, self.alpha, self.lora_dropout)
            block.router.w2 = LoRAWrapper(block.router.w2, self.lora_location['w2'], i, self.rank, self.alpha, self.lora_dropout)
            block.router.w3 = LoRAWrapper(block.router.w3, self.lora_location['w3'], i, self.rank, self.alpha, self.lora_dropout)
            
    def forward(self, x):
        return self.foundation_model(x)



