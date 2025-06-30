import torch.nn as nn
import torch
from attention import Attention_v1

class Multihead_Attention_Wrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            Attention_v1(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)