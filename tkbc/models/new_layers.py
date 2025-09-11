import torch 
import torch.nn as nn

class FourierMixer(nn.Module):
    """Simplified Fourier-based token mixer"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Simple learnable frequency filter
        self.freq_filter = nn.Parameter(torch.ones(dim // 2 + 1))
        
    def forward(self, x):
        # x: [batch_size, dim]
        # Apply 1D FFT
        x_fft = torch.fft.rfft(x, dim=-1)
        # Apply frequency filter
        x_fft = x_fft * self.freq_filter
        # Inverse FFT
        x_filtered = torch.fft.irfft(x_fft, n=self.dim, dim=-1)
        return x_filtered


class SimpleMetaBlock(nn.Module):
    """Simplified MetaFormer block"""
    def __init__(self, dim, use_fourier=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = FourierMixer(dim) if use_fourier else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x):
        # Token mixing
        x = x + self.mixer(self.norm1(x))
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x
