"""Masked Autoencoder for audio spectrograms (unified ViT-Base).

Architecture follows Audio-MAE (Huang et al., NeurIPS 2022): the encoder is a
ViT-Base over visible spectrogram patches only; a lightweight 4-block decoder
reconstructs the masked 16x16 patches. The masking strategy is fully decoupled
from the model: masks are computed outside (see ``eamcm.masking``) and passed
in as boolean tensors of shape (B, N), N = 512 for a 64x8 patch grid.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from .vit import Block, DecoderBlock, PatchEmbedAudio


class AudioMAE(nn.Module):
    """Audio MAE with pluggable masking.

    Args:
        img_size: (T, F) spectrogram size, default (1024, 128).
        patch_size: (16, 16) -> 64x8 grid, 512 tokens.
        in_chans: input channels (1 for log-Mel).
        decoder_embed_dim / decoder_depth / decoder_num_heads: lightweight
            asymmetric decoder, default 512 / 4 / 16 as in Audio-MAE.
        norm_pix_loss: compute the MSE loss on (per-patch) normalized targets,
            which stabilizes reconstruction of high-energy focal regions.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (1024, 128),
        patch_size: Tuple[int, int] = (16, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_num_heads: int = 16,
        norm_pix_loss: bool = True,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedAudio(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.encoder_embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # --- lightweight decoder ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        patch_out = patch_size[0] * patch_size[1] * in_chans
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_out, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.criterion = nn.MSELoss(reduction="none")

        self.initialize_weights()

    # ------------------------------------------------------------------ init
    def initialize_weights(self) -> None:
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        dec_pos = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=False
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dec_pos).float().unsqueeze(0))

        self.apply(self._init_module)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------- utilities
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, 1, T, F) -> (B, N, P) with P = 16*16."""
        p_t, p_f = self.patch_embed.patch_size
        b, c, t, f = imgs.shape
        imgs = imgs.reshape(b, c, t // p_t, p_t, f // p_f, p_f)
        imgs = imgs.permute(0, 2, 4, 3, 5, 1).reshape(b, -1, p_t * p_f * c)
        return imgs

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p_t, p_f = self.patch_embed.patch_size
        c = 1
        b, n = x.shape[:2]
        x = x.reshape(b, self.patch_embed.grid_size[0], self.patch_embed.grid_size[1], p_t, p_f, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, c, self.patch_embed.img_size[0], self.patch_embed.img_size[1])
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token", "pos_embed", "mask_token", "decoder_pos_embed"}

    # ---------------------------------------------------------------- encoder
    def forward_encoder(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T, F) normalized spectrogram; mask: (B, N) bool, True = masked."""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x = x[~mask.bool()].reshape(x.shape[0], -1, x.shape[-1])  # visible tokens only
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)
        return x

    # ---------------------------------------------------------------- decoder
    def forward_decoder(self, latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(latent[:, 1:, :])  # (B, V, D) visible tokens, original order
        b, _, d = x.shape
        n = mask.shape[1]
        # scatter visible embeddings into their original slots; mask tokens elsewhere
        x_full = self.mask_token.expand(b, n, d).clone()
        vis = (~mask.bool()).nonzero(as_tuple=False)  # (V*B, 2), row-major -> matches x order
        x_full[vis[:, 0], vis[:, 1]] = x.reshape(-1, d)
        x_full = x_full + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x_full, _ = blk(x_full)
        x_full = self.decoder_norm(x_full)
        return self.decoder_pred(x_full)

    # ------------------------------------------------------------------ loss
    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N)
        return (loss * mask.float()).sum() / mask.float().sum()

    def forward_per_patch_loss(
        self, imgs: torch.Tensor, mask: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Per-patch reconstruction MSE (B, N) used by hard-patch mining."""
        if targets is None:
            targets = imgs
        latent = self.forward_encoder(imgs, mask)
        pred = self.forward_decoder(latent, mask)
        target = self.patchify(targets)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        return ((pred - target) ** 2).mean(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Full MAE forward.

        Args:
            x: normalized spectrogram (B, 1, T, F).
            mask: (B, N) bool, True = masked.
            targets: optional raw (unnormalized) spectrogram used as
                reconstruction target; defaults to ``x``.
        Returns:
            (loss, pred, mask)
        """
        if targets is None:
            targets = x
        latent = self.forward_encoder(x, mask)
        pred = self.forward_decoder(latent, mask)
        loss = self.forward_loss(targets, pred, mask)
        return loss, pred, mask

    @torch.no_grad()
    def forward_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Full-spectrogram encoder pass with CLS attention rollout.

        Used by attention-guided teacher strategies (AttMask, Att-Guided).
        Returns per-patch attention (B, N) in [0, 1].
        """
        x = self.patch_embed(x) + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        rollout = torch.eye(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
        for blk in self.blocks:
            x, attn = blk(x, return_attn=True)  # attn: (B, heads, N+1, N+1)
            attn = attn.mean(dim=1)  # average heads
            attn = attn + torch.eye(attn.shape[-1], device=attn.device).unsqueeze(0)
            attn = attn / attn.sum(dim=-1, keepdim=True)
            rollout = attn @ rollout
        # CLS -> patch attention
        cls_attn = rollout[:, 0, 1:]  # (B, N)
        cls_attn = (cls_attn - cls_attn.amin(dim=1, keepdim=True)) / (
            cls_attn.amax(dim=1, keepdim=True) - cls_attn.amin(dim=1, keepdim=True) + 1e-8
        )
        return cls_attn


class AudioClassifier(nn.Module):
    """Fine-tuning head: encoder + global average pooling + linear classifier."""

    def __init__(
        self,
        encoder: AudioMAE,
        num_classes: int,
        multilabel: bool = False,
        global_pool: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.global_pool = global_pool
        self.multilabel = multilabel
        embed_dim = encoder.encoder_embed_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed[:, 1:, :]
        cls = (self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.encoder.blocks:
            x, _ = blk(x)
        x = self.encoder.norm(x)
        if self.global_pool:
            return x[:, 1:, :].mean(dim=1)  # GAP over patches
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def criterion(self, logits: torch.Tensor, targets: torch.Tensor):
        if self.multilabel:
            return nn.functional.binary_cross_entropy_with_logits(logits, targets.float())
        return nn.functional.cross_entropy(logits, targets, label_smoothing=0.1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size, cls_token: bool = False) -> np.ndarray:
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, -1])
    emb_h = _1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = _1d_sincos_pos_embed(embed_dim // 2, grid[1])
    pos = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos = np.concatenate([np.zeros([1, embed_dim]), pos], axis=0)
    return pos


def _1d_sincos_pos_embed(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", positions.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def load_audio_mae_pretrained(model: AudioMAE, checkpoint_path: str, strict: bool = False) -> None:
    """Initialize from an Audio-MAE checkpoint (AS -> P.A.S.T. two-stage SSL).

    Handles both raw ``state_dict`` checkpoints and the official Audio-MAE
    release format. Missing/unexpected keys are reported but tolerated when
    ``strict=False``.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    cleaned = {}
    for k, v in state.items():
        k = k.replace("module.", "").replace("student.", "").replace("backbone.", "")
        if k.startswith("encoder."):
            k = k[len("encoder.") :]
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    print(f"[load_audio_mae_pretrained] missing={len(missing)} unexpected={len(unexpected)}")


def build_audiomae(variant: str = "base", **overrides) -> AudioMAE:
    if variant != "base":
        raise ValueError(f"Unsupported variant: {variant}")
    kwargs = dict(
        img_size=(1024, 128),
        patch_size=(16, 16),
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=16,
        norm_pix_loss=True,
    )
    kwargs.update(overrides)
    return AudioMAE(**kwargs)
