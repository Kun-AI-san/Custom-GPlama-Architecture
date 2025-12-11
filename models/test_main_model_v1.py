import torch.nn as nn
import torch
from models.multihead_attention import Multihead_Attention, Grouped_Query_Attention, Multihead_Latent_Attention
from flash_attn.modules.mha import MHA

class LLM_v1(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_embedding = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(*[Transformer_Block(cfg) for _ in range(cfg['n_layers'])])
        self.final_norm = RMS_norm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, inp):
        batch_size, sequence_length = inp.shape
        token_embeddings = self.tok_embedding(inp)
        x = token_embeddings
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        out = self.out_head(x)
        return out

class RMS_norm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x).type_as(x)
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
            nn.Linear(cfg['emb_dim'], int(cfg['emb_dim'] * 3)),
            GELU(),
            nn.Linear(int(cfg['emb_dim'] * 3), cfg['emb_dim'])
        )
    
    def forward(self, x):
        return self.layers(x)
    
class FeedForward_SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg['emb_dim'], int(cfg['emb_dim'] * 3), bias=False)
        self.w2 = nn.Linear(int(3 * cfg['emb_dim']), cfg['emb_dim'], bias=False)
        self.w3 = nn.Linear(cfg['emb_dim'], int(cfg['emb_dim'] * 3), bias=False)
    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))
    
# class Router(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.router_linear = nn.Linear(cfg['emb_dim'], cfg['no_of_experts'])
#         self.noise_linear = nn.Linear(cfg['emb_dim'], cfg['no_of_experts'])
#         self.topk = int(cfg['no_of_experts']/4)

#     def forward(self, x):
#         router_out = self.router_linear(x)
#         if self.training:
#             noise_out = self.noise_linear(x)
#             noise = torch.randn_like(router_out) * nn.functional.softplus(noise_out)
#             router_out = router_out + noise
#         router_out_tk, router_index = torch.topk(router_out, self.topk)
#         top_experts = torch.softmax(router_out_tk, dim=-1)
#         return top_experts, router_index
    
# class MOE(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.router = Router(cfg)
#         self.experts = nn.ModuleList([FeedForward_SwiGLU(cfg) for i in range(cfg['no_of_experts'])])
    
#     def forward(self, x):
#         router_out, router_ind = self.router(x)

#         x_flat = x.view(-1, x.size(-1)) #B*T, emb_dim
#         router_out_flat = router_out.view(-1, router_out.size(-1)) #B*T, K
#         router_ind_flat = router_ind.view(-1, router_ind.size(-1)) #B*T, K
#         final_output_flat = torch.zeros_like(x_flat) #B*T, emb_dim
    
#         for i, expert in enumerate(self.experts):
#             gate_mask = (router_ind_flat == i)
#             if gate_mask.any():
#                 token_ind, topk_ind = torch.where(gate_mask)
#                 if len(token_ind) >0:
#                     gated_input = x_flat[token_ind]
#                     router_weights = router_out_flat[token_ind, topk_ind]
#                     expert_output = expert(gated_input)
#                     final_output_flat[token_ind] += expert_output * router_weights.unsqueeze(-1)
#         return final_output_flat.view(x.shape)
    
class Transformer_Block(nn.Module):

    def __init__(self, cfg:dict):
        super().__init__()
        self.layer_norm1 = RMS_norm(cfg['emb_dim'])
        if cfg['attention_type'] == 'gqa':
            if cfg['use_flash_attention']:
                self.multihead_attention = MHA(
                    embed_dim=cfg['emb_dim'],
                    num_heads=cfg['n_heads'],
                    dropout=0.0, 
                    use_flash_attn=True,
                    num_heads_kv=cfg['n_groups'],
                    rotary_emb_dim=(cfg['emb_dim']//cfg['n_heads']),
                    rotary_emb_interleaved=True,
                    causal=True
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
        elif cfg['attention_type'] == 'mha':
            if cfg['use_flash_attention']:
                self.multihead_attention = MHA(
                    embed_dim=cfg['emb_dim'],
                    num_heads=cfg['n_heads'],
                    dropout=0.0, 
                    use_flash_attn=True,
                    rotary_emb_dim=(cfg['emb_dim']//cfg['n_heads']),
                    rotary_emb_interleaved=True,
                    causal=True
                )
            else:
                self.multihead_attention = Multihead_Attention(
                d_in=cfg['emb_dim'], d_out=cfg['emb_dim'],
                context_length=cfg['context_length'],
                dropout=cfg['drop_rate'],
                num_heads=cfg['n_heads'],
                qkv_bias=cfg['qkv_bias'],
            )
        else:
            #work in progress
            self.multihead_attention = Multihead_Latent_Attention(
                d_in=cfg['emb_dim'], d_out=cfg['emb_dim'],
                context_length=cfg['context_length'],
                dropout=cfg['drop_rate'],
                num_heads=cfg['n_heads'],
                qkv_bias=cfg['qkv_bias'],
            )

        self.router = FeedForward_SwiGLU(cfg)
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
        x = self.router(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
