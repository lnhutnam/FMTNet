from typing import Tuple, Sequence
from functools import partial, reduce
import torch
from torch import nn
import torch.nn.functional as F

from models.bases import TKBCModel

import math


class FourierTransform(nn.Module):
    """
    Fourier Transform token mixer that operates in frequency domain
    """
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.dim = dim
        # Learnable scaling factors for frequency components
        self.freq_weights = nn.Parameter(torch.ones(dim // 2 + 1, dtype=torch.float32))
        # Optional learnable phase shifts
        self.phase_shifts = nn.Parameter(torch.zeros(dim // 2 + 1, dtype=torch.float32))
        
    def forward(self, x):
        """
        Apply Fourier transform token mixing
        Input: x with shape [B, C, H, W] or [B, N, C]
        """
        original_shape = x.shape
        
        # Handle different input shapes
        if len(original_shape) == 4:
            B, C, H, W = original_shape
            x = x.view(B, C, H * W).transpose(-1, -2)  # [B, N, C]
        elif len(original_shape) == 3:
            B, N, C = original_shape
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        
        # Apply FFT along the sequence dimension (token dimension)
        x_fft = torch.fft.rfft(x, dim=1)  # [B, N//2+1, C]
        
        # Apply learnable frequency weighting and phase shifts
        freq_weights = self.freq_weights.unsqueeze(0).unsqueeze(-1)  # [1, N//2+1, 1]
        phase_shifts = self.phase_shifts.unsqueeze(0).unsqueeze(-1)  # [1, N//2+1, 1]
        
        # Apply magnitude scaling and phase adjustment
        magnitude = torch.abs(x_fft) * freq_weights
        phase = torch.angle(x_fft) + phase_shifts
        x_fft_modified = magnitude * torch.exp(1j * phase)
        
        # Inverse FFT to get back to spatial domain
        x_reconstructed = torch.fft.irfft(x_fft_modified, n=x.shape[1], dim=1)
        
        # Reshape back to original format
        if len(original_shape) == 4:
            x_reconstructed = x_reconstructed.transpose(-1, -2).view(B, C, H, W)
        
        return x_reconstructed


class LayerNormChannel(nn.Module):
    """
    LayerNorm for channels-first data format
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        if len(x.shape) == 4:  # [B, C, H, W]
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        else:  # [B, N, C]
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = x * self.weight + self.bias
        return x


class Mlp(nn.Module):
    """
    MLP module for MetaFormer blocks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # Handle channel-first format
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class MetaFormerBlock(nn.Module):
    """
    MetaFormer block with Fourier Transform token mixer
    """
    def __init__(self, dim, 
                 token_mixer=FourierTransform, 
                 mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=LayerNormChannel, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            if len(x.shape) == 4:  # [B, C, H, W]
                x = x + self.drop_path(
                    self.layer_scale_1.view(1, -1, 1, 1) * self.token_mixer(self.norm1(x)))
                x = x + self.drop_path(
                    self.layer_scale_2.view(1, -1, 1, 1) * self.mlp(self.norm2(x)))
            else:  # [B, N, C]
                x = x + self.drop_path(
                    self.layer_scale_1 * self.token_mixer(self.norm1(x)))
                x = x + self.drop_path(
                    self.layer_scale_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TNTComplExMetaFormer(TKBCModel):
    """
    TNTComplEx model enhanced with MetaFormer architecture using Fourier Transform token mixing
    """
    def __init__(
        self,
        sizes: Tuple[int, int, int, int],
        rank: int,
        no_time_emb=False,
        init_size: float = 1e-2,
        num_metaformer_layers: int = 2,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
    ):
        super(TNTComplExMetaFormer, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.no_time_emb = no_time_emb

        # Original TNTComplEx embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1]]  # entities, relations, time, no_time_relations
        ])
        
        # Initialize embeddings
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

        # MetaFormer layers for enhanced representation learning
        self.metaformer_layers = nn.ModuleList()
        
        # Create MetaFormer blocks
        for i in range(num_metaformer_layers):
            dpr = drop_path_rate * i / (num_metaformer_layers - 1) if num_metaformer_layers > 1 else 0.
            block = MetaFormerBlock(
                dim=2 * rank,  # Same as embedding dimension
                token_mixer=FourierTransform,
                mlp_ratio=mlp_ratio,
                drop_path=dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value
            )
            self.metaformer_layers.append(block)

        # Layer normalization for final output
        self.final_norm = LayerNormChannel(2 * rank)

    @staticmethod
    def has_time():
        return True

    def enhance_embeddings(self, embeddings):
        """
        Apply MetaFormer enhancement to embeddings
        Input: embeddings [B, C] -> reshape to [B, C, 1, 1] for MetaFormer processing
        """
        B, C = embeddings.shape
        # Reshape to 4D for MetaFormer processing
        x = embeddings.view(B, C, 1, 1)
        
        # Apply MetaFormer blocks
        for layer in self.metaformer_layers:
            x = layer(x)
        
        # Apply final normalization and reshape back
        x = self.final_norm(x)
        return x.view(B, C)

    def score(self, x):
        # Get embeddings
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        # Enhance embeddings with MetaFormer
        lhs = self.enhance_embeddings(lhs)
        rel = self.enhance_embeddings(rel)
        rel_no_time = self.enhance_embeddings(rel_no_time)
        rhs = self.enhance_embeddings(rhs)
        time = self.enhance_embeddings(time)

        # Split into real and imaginary parts (original TNTComplEx logic)
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        # TNTComplEx relation-time interaction
        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        # TNTComplEx scoring function
        return torch.sum(
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0] +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        # Get embeddings
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        # Enhance embeddings with MetaFormer
        lhs = self.enhance_embeddings(lhs)
        rel = self.enhance_embeddings(rel)
        rel_no_time = self.enhance_embeddings(rel_no_time)
        rhs = self.enhance_embeddings(rhs)
        time = self.enhance_embeddings(time)

        # Split into real and imaginary parts
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        # Get all entity embeddings and enhance them
        right = self.embeddings[0].weight
        enhanced_right = self.enhance_embeddings(right)
        right = enhanced_right[:, :self.rank], enhanced_right[:, self.rank:]

        # TNTComplEx relation-time computation
        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]

        # TNTComplEx regularizer (original logic)
        regularizer = (
           math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
           torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
           torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
           math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

        # Compute scores
        scores = (
               (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
               (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
        )

        # Time embeddings for regularization
        time_emb = self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

        return scores, regularizer, time_emb

    def forward_over_time(self, x):
        # Get embeddings
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight
        rel_no_time = self.embeddings[3](x[:, 1])

        # Enhance embeddings
        lhs = self.enhance_embeddings(lhs)
        rel = self.enhance_embeddings(rel)
        rhs = self.enhance_embeddings(rhs)
        time = self.enhance_embeddings(time)
        rel_no_time = self.enhance_embeddings(rel_no_time)

        # Split into real and imaginary parts
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        # TNTComplEx forward over time computation
        score_time = (
            (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
             lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
            (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
             lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        rhs_embeddings = self.embeddings[0].weight.data[chunk_begin:chunk_begin + chunk_size]
        # Enhance the chunk
        enhanced_rhs = self.enhance_embeddings(rhs_embeddings)
        return enhanced_rhs.transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        # Get embeddings
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rel_no_time = self.embeddings[3](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])

        # Enhance embeddings
        lhs = self.enhance_embeddings(lhs)
        rel = self.enhance_embeddings(rel)
        rel_no_time = self.enhance_embeddings(rel_no_time)
        time = self.enhance_embeddings(time)

        # Split into real and imaginary parts
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        # TNTComplEx query computation
        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        return torch.cat([
            lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
            lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
        ], 1)