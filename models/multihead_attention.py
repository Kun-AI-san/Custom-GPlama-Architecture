import torch.nn as nn
import torch

def precompute_freqs(theta=10000, end=2048, emb_dim=896):
    freqs = 1/(theta ** (torch.arange(0, emb_dim, 2, requires_grad=False, device='cuda')/emb_dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    cs = torch.outer(t, freqs)
    pre_freqs = torch.polar(torch.ones_like(cs), cs)
    return pre_freqs

def apply_rotatory_emb(q, k, pre_freqs, d_in):
    pre_freqs = pre_freqs.unsqueeze(0).unsqueeze(0)
    q_ = torch.view_as_complex(q.view(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.view(*k.shape[:-1], -1, 2))
    q_out = torch.view_as_real(q_ * pre_freqs).flatten(3)
    k_out = torch.view_as_real(k_ * pre_freqs).flatten(3)
    return q_out.type_as(q), k_out.type_as(k)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Multihead_Attention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out%num_heads==0), \
        "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out//num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        vals = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        vals = vals.view(b, num_tokens, self.num_heads, self.head_dim)

        pre_freqs = precompute_freqs(end=num_tokens)
        queries, keys = apply_rotatory_emb(queries.view(b, num_tokens, d_in), keys.view(b, num_tokens, d_in), pre_freqs, d_in)

        queries, keys = queries.view(b, num_tokens, self.num_heads, self.head_dim), keys.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        vals = vals.transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)

        masked_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores.masked_fill_(masked_bool, -torch.inf)

        attention_weights = torch.softmax(attention_scores/(keys.shape[-1]**0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vec = (attention_weights@vals).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
    

class Grouped_Query_Attention(nn.Module):

    def __init__(self, d_in, d_out, n_groups, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out%num_heads==0), \
        "d_out must be divisible by num_heads"
        assert(d_out%n_groups==0), \
        "d_out must be divisible by n_groups"
        assert(num_heads%n_groups==0), \
        "num_heads must be divisible by n_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out//num_heads
        self.n_groups = n_groups
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out//(num_heads//n_groups), bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out//(num_heads//n_groups), bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        vals = self.W_value(x)
        
        keys = keys.view(b, num_tokens, self.n_groups, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        vals = vals.view(b, num_tokens, self.n_groups, self.head_dim)
        
        keys = repeat_kv(keys, n_rep=(self.num_heads//self.n_groups))
        vals = repeat_kv(vals, n_rep=(self.num_heads//self.n_groups))

        pre_freqs = precompute_freqs(end=num_tokens)
        queries, keys = apply_rotatory_emb(queries.view(b, num_tokens, d_in), keys.view(b, num_tokens, d_in), pre_freqs)

        queries, keys = queries.view(b, num_tokens, self.num_heads, self.head_dim), keys.view(b, num_tokens, self.num_heads, self.head_dim)
        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        vals = vals.transpose(1, 2)
        
        attention_scores = queries @ keys.transpose(2, 3)
        
        masked_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores.masked_fill_(masked_bool, -torch.inf)

        attention_weights = torch.softmax(attention_scores/(keys.shape[-1]**0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)
        context_vec = (attention_weights@vals).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
    

class Multihead_Latent_Attention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        # In Progress
        super().__init__()
        assert (d_out%num_heads==0), \
        "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out//num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_kv = nn.Linear(d_in, int(d_out/3), bias=qkv_bias)
        self.W_uk = nn.Linear(int(d_in/3), d_out, bias=qkv_bias)
        self.W_uv = nn.Linear(int(d_in/3), d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(int(d_out/3))
        self.register_buffer(
            'absorbed_k', None
        )

    def forward(self, x, kv_cache=None, past_length=0):
        b, num_tokens, d_in = x.shape
        
        if self.absorbed_k is None:
            absorbed = torch.matmul(self.W_query.weight, self.W_uk.weight)
            self.absorbed_k = absorbed.view(self.num_heads, self.head_dim, -1)

        new_c_kv = self.ln(self.W_kv(x))
        if kv_cache is None:
            c_kv = new_c_kv
        else:
            c_kv = torch.cat([kv_cache, new_c_kv], dim=1)
        
        S_full = c_kv.size(1)

        v_full = self.W_uv(c_kv)
        v = v_full.view(b, S_full, self.num_heads, self.head_dim).transpose(1, 2)

        q = x.view(b, num_tokens, self.num_heads, self.head_dim)

        attention_scores = torch.zeros(b, self.num_heads, num_tokens, S_full, device=x.device)

        for h in range(self.num_heads):
            tmp = torch.matmul(q[:, :, h], self.absorbed_k[h])
            attention_scores[:, h] = torch.bmm(tmp, c_kv.transpose(1, 2))

        attention_scores = attention_scores/(self.head_dim**0.5)
        mask = torch.tril(torch.ones((num_tokens, S_full), device=x.device), diagonal=past_length)
        attention_scores = attention_scores.masked_fill(mask.view(1, 1, num_tokens, S_full) == 0, -torch.inf)

        attention_weights = torch.softmax(attention_scores, dim=-1)

        out_heads = []
        for h in range(self.num_heads):
            context_h = torch.matmul(attention_weights[:, h], v[:, h])
            out_heads.append(context_h)

        out = torch.cat(out_heads, dim=-1)
        context_vec = self.out_proj(out)
        return context_vec, c_kv