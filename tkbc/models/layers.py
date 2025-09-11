import torch
from torch import nn
import torch.nn.functional as F

class FourierTokenMixer(nn.Module):
    """Token mixer using Fourier Transform for frequency domain processing"""
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.dim = dim
        # Learnable frequency domain weights
        self.freq_weights = nn.Parameter(torch.ones(dim // 2 + 1, dtype=torch.complex64))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: [B, C, H, W] or [B, N, C]
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x_flat = x.view(B, C, -1)  # [B, C, N]
        else:
            x_flat = x.transpose(1, 2)  # [B, C, N]
            
        # Apply FFT along the spatial dimension
        x_fft = torch.fft.rfft(x_flat, dim=-1)
        
        # Apply learnable weights in frequency domain
        x_fft = x_fft * self.freq_weights.unsqueeze(0).unsqueeze(0)
        
        # Apply inverse FFT
        x_filtered = torch.fft.irfft(x_fft, n=x_flat.shape[-1], dim=-1)
        
        # Apply dropout
        x_filtered = self.dropout(x_filtered)
        
        # Reshape back to original shape
        if len(x.shape) == 4:
            return x_filtered.view(B, C, H, W)
        else:
            return x_filtered.transpose(1, 2)


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


class LayerNormChannel(nn.Module):
    """Channel-wise LayerNorm for [B, C, H, W] tensors"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class Mlp(nn.Module):
    """MLP module"""
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
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MetaFormerBlock(nn.Module):
    """MetaFormer block with token mixer and MLP"""
    def __init__(self, dim, token_mixer=nn.Identity, mlp_ratio=4., 
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
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * 
                self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * 
                self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class EmbeddingStacker(nn.Module):
    """Stack entity, relation, and time embeddings for MetaFormer input"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.projection = nn.Linear(embed_dim * 4, embed_dim)  # lhs + rel + rhs + time
        
    def forward(self, lhs_emb, rel_emb, rhs_emb, time_emb):
        # Stack embeddings: [batch_size, 4 * embed_dim]
        stacked = torch.cat([lhs_emb, rel_emb, rhs_emb, time_emb], dim=-1)
        # Project to target dimension
        projected = self.projection(stacked)
        return projected