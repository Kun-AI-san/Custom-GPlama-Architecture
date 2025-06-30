import torch.nn as nn
import torch
from models.multihead_attention import Multihead_Attention, Grouped_Query_Attention, Multihead_Latent_Attention
from flash_attn.modules.mha import MHA

class LLM_v1(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_embedding = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_embedding = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(*[Transformer_Block(cfg) for _ in range(cfg['n_layers'])])
        self.final_norm = LayerNormalization(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, inp):
        batch_size, sequence_length = inp.shape
        token_embeddings = self.tok_embedding(inp)
        positional_embeddings = self.pos_embedding(torch.arange(sequence_length, device=inp.device))
        x = token_embeddings + positional_embeddings
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        out = self.out_head(x)
        return out

class RMS_norm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LayerNormalization(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x:torch.tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean)/torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], int(cfg['emb_dim'] * 2)),
            GELU(),
            nn.Linear(int(cfg['emb_dim'] * 2), cfg['emb_dim'])
        )
    
    def forward(self, x):
        return self.layers(x)
    
class FeedForward_SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg['emb_dim'], int(cfg['emb_dim'] * 1.8), bias=False)
        self.w2 = nn.Linear(int(1.8 * cfg['emb_dim']), cfg['emb_dim'], bias=False)
        self.w3 = nn.Linear(cfg['emb_dim'], int(cfg['emb_dim'] * 1.8), bias=False)
    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))
    
class Transformer_Block(nn.Module):

    def __init__(self, cfg:dict):
        super().__init__()
        self.layer_norm1 = RMS_norm(cfg['emb_dim'])
        if cfg['use_flash_attention']:
            self.multihead_attention = MHA(
                embed_dim=cfg['emb_dim'],
                num_heads=cfg['n_heads'],
                dropout=0.0, 
                use_flash_attn=True,
            )
        else:
            self.multihead_attention = Grouped_Query_Attention(
            d_in=cfg['emb_dim'], d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            dropout=cfg['drop_rate'],
            num_heads=cfg['n_heads'],
            qkv_bias=cfg['qkv_bias'],
            n_groups=cfg['n_groups'],
        )
        self.feed_forward = FeedForward_SwiGLU(cfg)
        self.layer_norm2 = RMS_norm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])
    
    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)

        x = self.multihead_attention(x)

        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
