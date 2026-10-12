"""Vision Transformer building blocks (ViT-Base) for audio MAE.

Implements the standard MAE components (patch embedding with convolutional
stem, pre-norm Transformer blocks, lightweight decoder blocks) following
He et al. 2022 and the audio instantiation of Huang et al. 2022 (Audio-MAE).
"""

from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PatchEmbedAudio(nn.Module):
    """Split a (T=1024, F=128) spectrogram into non-overlapping 16x16 patches.

    Yields a 64 x 8 patch grid (512 tokens) that preserves symmetric 2D
    structural integrity (see the patch-size ablation in our supplement:
    16x16 consistently outperforms the stretched 32x8 setting).
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (1024, 128),
        patch_size: Tuple[int, int] = (16, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_dropped = self.attn_drop(attn)
        out = (attn_dropped @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        if return_attn:
            return out, attn
        return out, None


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int, act_layer=nn.GELU, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6)) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        y, attn = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        if return_attn:
            return x, attn
        return x, None


def DecoderBlock(
    dim: int, num_heads: int, mlp_ratio: float = 4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6)
) -> Block:
    return Block(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
