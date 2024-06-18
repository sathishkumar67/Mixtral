import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from functions import precompute_theta_pos_frequencies, apply_rotary_embeddings, repeat_kv
from moe import SparseMoE
from xformers.ops.fmha.attn_bias import LocalAttentionFromBottomRightMask


from pathlib import Path
from sentencepiece import SentencePieceProcessor
from typing import List



class RMSNorm(nn.Module):
    """Computes Root Mean Squared Normalization along the Layer Dimension, this Normalization avoids computing unwanted computations.
    \nParams:
        dim: hidden_size dimension or embedding size
        eps: a very small number to avoid dividing the denominator by zero
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        # A small no to avoid dividing by zero
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x) 
    


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.n_kv_heads = args.n_kv_heads
        self.n_q_heads = args.n_heads
        self.n_rep = self.n_q_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        bias = LocalAttentionFromBottomRightMask(window_left=self.args.sliding_window_size, window_right=0)
        local_attention = bias.materialize(shape=(seq_len, seq_len)).to(self.args.device)
        scores.masked_fill(local_attention==-torch.inf, -torch.inf)
        scores = self.dropout(F.softmax(scores.float(), dim=-1).type_as(xq))

        output = torch.matmul(scores, xv)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        
        return self.dropout(self.wo(output))
    


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = Attention(args)
        self.smoe = SparseMoE(args)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.smoe_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor):
        x = x + self.attention.forward(self.attention_norm(x), freqs_complex)
        x = x + self.smoe.forward(self.smoe_norm(x))

        return x
    

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.token_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(Block(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len, device=self.args.device)

    def forward(self, x: torch.Tensor):
        # Multiply the embedded input with sqrt(dim) to scale the embeddings
        x = (self.token_embeddings(x) * math.sqrt(self.args.dim))
        
        for layer in self.layers:
            x = layer(x, self.freqs_complex)
        
        return self.output(self.norm(x))