import math
import random
from torch import nn, einsum
import numpy as np
import torch
from einops import rearrange, repeat
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def total_variation(trajectory):
    """
    计算轨迹数据的总变差。
    
    参数:
        trajectory (torch.Tensor): 输入轨迹数据，形状为 (B, 60, 10)。
        
    返回:
        total_variation (torch.Tensor): 每个样本每个特征的总变差，形状为 (B, 10)。
    """
    diff = torch.diff(trajectory, dim=1)  # 形状: (B, 59, 10)
    total_variation = torch.sum(torch.abs(diff), dim=1)  # 形状: (B, 10)
    return total_variation


def curve_smoothing(all_mars):
    b, f, d = all_mars.shape
    all_mars = all_mars.permute(0, 2, 1).reshape(-1, 1, f)
    kernel_size = 3
    kernel = (torch.ones(1, 1, kernel_size) / kernel_size).to(all_mars.device)
    padding = kernel_size // 2
    all_mars_padded = F.pad(all_mars, (padding, padding), mode='replicate')
    smoothed_all_mars = F.conv1d(all_mars_padded, kernel)
    return smoothed_all_mars.reshape(b, d, f).permute(0, 2, 1)


def jitter_regularization(trajectory):
    diff1 = trajectory[:, 1:] - trajectory[:, :-1]
    diff2 = diff1[:, 1:] - diff1[:, :-1]
    return torch.mean(diff2**2)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

def default(val, d):
    if val is not None:
        return val
    return d



class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_position=2048):
        super(RotaryPositionEmbedding, self).__init__()
        if dim % 2 != 0:
            raise ValueError("Dimension must be even for RoPE.")
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x shape: (batch_size * num_heads, seq_len, head_dim)
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", position, self.inv_freq)
        sin = sinusoid_inp.sin().unsqueeze(0)  # (1, seq_len, head_dim // 2)
        cos = sinusoid_inp.cos().unsqueeze(0)  # (1, seq_len, head_dim // 2)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
        return x_rotated


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., block_size=60, rope=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.attn_dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, block_size, block_size))
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.rotary_emb = RotaryPositionEmbedding(dim_head) if rope else None

    def forward(self, x, context=None, mask=False):
        B, T, C = x.size()
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # 应用旋转位置编码
        if self.rotary_emb is not None:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask:
            sim = sim.masked_fill(self.bias[:,:T,:T] == 0, float('-inf'))

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

if __name__ == '__main__':
    batch_size, num_heads, seq_len, head_dim = 2, 8, 10, 64
    q = torch.randn(batch_size*num_heads, seq_len, head_dim)
    k = torch.randn(batch_size*num_heads, seq_len, head_dim)
    rope = RotaryPositionEmbedding(head_dim)
    q_rot = rope(q)
    print(q_rot.shape)
