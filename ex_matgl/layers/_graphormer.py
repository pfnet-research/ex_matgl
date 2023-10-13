"""
Self Attention code is based on graphormer implementation
https://github.com/microsoft/Graphormer/blob/main/graphormer/models/graphormer_3d.py
"""
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


@torch.jit.script
def softmax_dropout(input, dropout_prob: float, is_training: bool):
    return F.dropout(F.softmax(input, -1), dropout_prob, is_training)


class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj: Callable[[Tensor], Tensor] = nn.Linear(
            embed_dim, embed_dim * 3, bias=bias
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: Tensor,
        batch_num_nodes: Tensor,
        attn_bias: Tensor = None,
    ) -> Tensor:
        n_node, embed_dim = query.size()
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        _shape = (n_node, self.num_heads, self.head_dim)
        q = q.contiguous().view(_shape).transpose(0, 1) * self.scaling
        k = k.contiguous().view(_shape).transpose(0, 1)
        v = v.contiguous().view(_shape).transpose(0, 1)
        # (batch x n_node, num_heads, head_dim)
        # -> (num_heads, batch x n_node, head_dim)

        attn_weights = torch.bmm(q, k.transpose(1,2))
        # (num_heads, batch x n_node, head_dim) x (num_heads, head_dim, batch x n_node)
        # -> (num_heads, batch x n_node, batch x n_node)
        
        blocks = [torch.ones(node_size,node_size, dtype=torch.float) for node_size in batch_num_nodes]
        batch_mask = torch.block_diag(*blocks)
        batch_mask = torch.unsqueeze(batch_mask,0)
        # (batch x n_node, batch x n_node)
        # -> (1, batch x n_node, batch x n_node)
        batch_mask2 = torch.nan_to_num((torch.ones_like(batch_mask,dtype=torch.float)-batch_mask)*(-sys.float_info.max))
        attn_weights = attn_weights*batch_mask+batch_mask2
        attn_probs = softmax_dropout(attn_weights, self.dropout, self.training)

        attn = torch.bmm(attn_probs, v)
        # (num_heads, batch x n_node, batch x n_node) x (num_heads, batch x n_node, batch x n_head)
        # -> (num_heads, batch x n_node, head_dim)

        attn = attn.transpose(0, 1).contiguous().view(n_node, embed_dim)
        # (num_heads, batch x n_node, head_dim)
        # -> (batch x n_node, num_heads, head_dim)
        # -> (batch x n_node, embed_dim)
        
        # attn = self.out_proj(attn)
        # (batch x n_node, embed_dim)
        # -> (batch x n_node, embed_dim)
        return attn