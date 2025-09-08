from typing import Tuple, Sequence
from functools import partial, reduce
import torch
from torch import nn
import torch.nn.functional as F

from models.bases import TKBCModel


class FourierTransform(nn.Module):
    """
    Fourier Transform token mixer with integrated MLP for complex numbers
    Performs FFT -> frequency domain processing -> MLP (complex numbers) -> inverse FFT
    """
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, drop=0., **kwargs):
        super().__init__()
        self.dim = dim
        
        # Learnable scaling factors for frequency components
        # self.freq_weights = nn.Parameter(torch.ones(dim // 2 + 1, dtype=torch.float32))
        # Learnable phase shifts
        # self.phase_shifts = nn.Parameter(torch.zeros(dim // 2 + 1, dtype=torch.float32))
        
        # MLP for complex numbers in frequency domain
        mlp_hidden_dim = int(dim * mlp_ratio)
        # Separate linear layers for real and imaginary parts
        self.fc1_real = nn.Linear(dim, mlp_hidden_dim)
        self.fc1_imag = nn.Linear(dim, mlp_hidden_dim)
        self.act = act_layer()
        self.fc2_real = nn.Linear(mlp_hidden_dim, dim)
        self.fc2_imag = nn.Linear(mlp_hidden_dim, dim)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        """
        Apply Fourier transform token mixing followed by MLP for complex numbers
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
        
        # Store original sequence length for inverse FFT
        seq_length = x.shape[1]
        
        # Apply FFT along the sequence dimension (token dimension)
        x_fft = torch.fft.rfft(x, dim=1)  # [B, N//2+1, C]
        
        # Apply learnable frequency weighting and phase shifts
        # freq_weights = self.freq_weights.unsqueeze(0).unsqueeze(-1)  # [1, N//2+1, 1]
        # phase_shifts = self.phase_shifts.unsqueeze(0).unsqueeze(-1)  # [1, N//2+1, 1]
        
        # Apply magnitude scaling and phase adjustment
        # magnitude = torch.abs(x_fft) * freq_weights
        # phase = torch.angle(x_fft) + phase_shifts
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        x_fft_modified = magnitude * torch.exp(1j * phase)
        
        # Split into real and imaginary parts for MLP processing
        x_real = x_fft_modified.real  # [B, N//2+1, C]
        x_imag = x_fft_modified.imag  # [B, N//2+1, C]
        
        # Apply MLP to real and imaginary parts separately
        x_real_mlp = self.fc1_real(x_real)
        x_imag_mlp = self.fc1_imag(x_imag)
        x_real_mlp = self.act(x_real_mlp)
        x_imag_mlp = self.act(x_imag_mlp)
        x_real_mlp = self.drop(x_real_mlp)
        x_imag_mlp = self.drop(x_imag_mlp)
        x_real_mlp = self.fc2_real(x_real_mlp)
        x_imag_mlp = self.fc2_imag(x_imag_mlp)
        x_real_mlp = self.drop(x_real_mlp)
        x_imag_mlp = self.drop(x_imag_mlp)
        
        # Recombine real and imaginary parts into complex numbers
        x_mlp = torch.complex(x_real_mlp, x_imag_mlp)  # [B, N//2+1, C]
        
        # Inverse FFT to get back to spatial domain
        x_reconstructed = torch.fft.irfft(x_mlp, n=seq_length, dim=1)
        
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
    Simplified MetaFormer block with integrated Fourier Transform + MLP
    """
    def __init__(self, dim, 
                 token_mixer=FourierTransform, 
                 mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=LayerNormChannel, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm = norm_layer(dim)
        self.fourier_mlp = token_mixer(
            dim=dim, 
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            if len(x.shape) == 4:  # [B, C, H, W]
                x = x + self.drop_path(
                    self.layer_scale.view(1, -1, 1, 1) * self.fourier_mlp(self.norm(x)))
            else:  # [B, N, C]
                x = x + self.drop_path(
                    self.layer_scale * self.fourier_mlp(self.norm(x)))
        else:
            x = x + self.drop_path(self.fourier_mlp(self.norm(x)))
        
        return x


class TPComplExMetaFormer(TKBCModel):
    """
    TPComplEx model enhanced with MetaFormer architecture using Fourier Transform token mixing
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
        super(TPComplExMetaFormer, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.no_time_emb = no_time_emb

        # Original TPComplEx embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], 2 * rank, sparse=True),  # entities
            nn.Embedding(sizes[1], 2 * rank, sparse=True),  # relations
            nn.Embedding(sizes[3], 6 * rank, sparse=True),  # time
        ])
        
        # Initialize embeddings
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        # MetaFormer layers for enhanced representation learning
        self.metaformer_layers = nn.ModuleList()
        
        # Create MetaFormer blocks
        for i in range(num_metaformer_layers):
            dpr = drop_path_rate * i / (num_metaformer_layers - 1) if num_metaformer_layers > 1 else 0.
            block = MetaFormerBlock(
                dim=2 * rank,  # Same as entity/relation embedding dimension
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
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        # Enhance embeddings with MetaFormer
        lhs = lhs + self.enhance_embeddings(lhs)
        rel = rel + self.enhance_embeddings(rel)
        rhs = rhs + self.enhance_embeddings(rhs)

        # Apply temporal transformations (original TPComplEx logic)
        lhs = (
            lhs[:, : self.rank] + torch.cos(time[:, 2 * self.rank : 3 * self.rank]),
            lhs[:, self.rank :] + torch.sin(time[:, 3 * self.rank : 4 * self.rank]),
        )

        rhs = (
            rhs[:, : self.rank] + torch.cos(time[:, 4 * self.rank : 5 * self.rank]),
            rhs[:, self.rank :] + torch.sin(time[:, 5 * self.rank : 6 * self.rank]),
        )

        rel = rel[:, : self.rank], rel[:, self.rank : 2 * self.rank]
        time = (
            time[:, : self.rank] + torch.cos(time[:, : self.rank]), 
            time[:, self.rank : 2 * self.rank] + torch.sin(time[:, self.rank : 2 * self.rank])
        )

        # Compute relation-time interaction
        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]), (rt[1] + rt[2])

        # Compute final score
        return torch.sum(
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0]
            + (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1],
            1,
            keepdim=True,
        )

    def forward(self, x):
        # Get embeddings
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        # Enhance embeddings with MetaFormer
        lhs = lhs + self.enhance_embeddings(lhs)
        rel = rel + self.enhance_embeddings(rel)
        rhs = rhs + self.enhance_embeddings(rhs)

        # Apply temporal transformations
        lhs = (
            lhs[:, : self.rank] + time[:, 2 * self.rank : 3 * self.rank],
            lhs[:, self.rank :] + time[:, 3 * self.rank : 4 * self.rank],
        )
        rhs = (
            rhs[:, : self.rank] + time[:, 4 * self.rank : 5 * self.rank],
            rhs[:, self.rank :] + time[:, 5 * self.rank : 6 * self.rank],
        )
        
        bias_t_r = time[:, 4 * self.rank : 5 * self.rank]
        bias_t_i = time[:, 5 * self.rank : 6 * self.rank]
        time = time[:, : self.rank], time[:, self.rank : 2 * self.rank]

        # Get all entity embeddings and enhance them
        right = self.embeddings[0].weight
        # Enhance all entity embeddings in batch
        if right.shape[0] > 1000:  # For large vocabularies, process in chunks
            enhanced_right_parts = []
            chunk_size = 1000
            for i in range(0, right.shape[0], chunk_size):
                chunk = right[i:i+chunk_size]
                enhanced_chunk = self.enhance_embeddings(chunk)
                enhanced_right_parts.append(enhanced_chunk)
            enhanced_right = torch.cat(enhanced_right_parts, dim=0)
        else:
            enhanced_right = self.enhance_embeddings(right)
        
        right = enhanced_right[:, : self.rank], enhanced_right[:, self.rank :]
        rel = rel[:, : self.rank], rel[:, self.rank : 2 * self.rank]

        # Compute relation-time interaction
        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        # Compute scores
        scores = (
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t()
            + torch.sum((lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * bias_t_r, 1, keepdim=True)
            + (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
            + torch.sum((lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * bias_t_i, 1, keepdim=True)
        )

        # Compute regularization terms
        factors = (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
        )

        # Time embeddings for regularization
        time_emb = (
            self.embeddings[2].weight[:-1] if self.no_time_emb 
            else self.embeddings[2].weight
        )

        return scores, factors, time_emb

    def forward_over_time(self, x):
        raise NotImplementedError("Forward over time not implemented for MetaFormer version.")

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        rhs_embeddings = self.embeddings[0].weight.data[chunk_begin : chunk_begin + chunk_size]
        # Enhance the chunk
        enhanced_rhs = self.enhance_embeddings(rhs_embeddings)
        return enhanced_rhs.transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        # Get embeddings
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])

        # Enhance embeddings
        lhs = lhs + self.enhance_embeddings(lhs)
        rel = rel + self.enhance_embeddings(rel)

        # Apply temporal transformations
        lhs = (
            lhs[:, : self.rank] + time[:, 2 * self.rank : 3 * self.rank],
            lhs[:, self.rank :] + time[:, 3 * self.rank : 4 * self.rank],
        )
        rel = rel[:, : self.rank], rel[:, self.rank : 2 * self.rank]
        time = time[:, : self.rank], time[:, self.rank : 2 * self.rank]

        # Compute relation-time interaction
        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]), (rt[1] + rt[2])

        return torch.cat([
            lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
            lhs[1] * full_rel[0] + lhs[0] * full_rel[1],
        ], 1)