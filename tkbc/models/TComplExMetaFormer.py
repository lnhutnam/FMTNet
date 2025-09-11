from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F

from models.bases import TKBCModel
from models.new_layers import FourierMixer, SimpleMetaBlock


class TComplExMetaFormer(TKBCModel):
    """Simplified TComplEx with MetaFormer enhancement"""
    
    def __init__(self, sizes: Tuple[int, int, int, int], rank: int,
                 num_blocks=1, use_fourier=True,
                 no_time_emb=False, init_size: float = 1e-2):
        super().__init__()
        self.sizes = sizes
        self.rank = rank
        self.no_time_emb = no_time_emb
        
        # Original TComplEx embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]]  # entities, relations, time
        ])
        
        # Initialize embeddings
        for emb in self.embeddings:
            emb.weight.data *= init_size
        
        # Simple embedding fusion
        self.fusion = nn.Linear(2 * rank * 3, 2 * rank)  # Combine all 4 embeddings
        
        # MetaFormer blocks
        self.meta_blocks = nn.ModuleList([
            SimpleMetaBlock(2 * rank, use_fourier) 
            for _ in range(num_blocks)
        ])
        
        # Output normalization
        self.output_norm = nn.LayerNorm(2 * rank)

        self.alpha = 0.75

    @staticmethod
    def has_time():
        return True

    def enhance_embeddings(self, lhs, rel, time):
        """Apply MetaFormer enhancement to embeddings"""
        # Stack all embeddings
        stacked = torch.cat([lhs, rel, time], dim=-1)  # [B, 8*rank]
        
        # Fuse to original dimension
        fused = self.fusion(stacked)  # [B, 2*rank]
        
        # Apply MetaFormer blocks
        enhanced = fused
        for block in self.meta_blocks:
            enhanced = block(enhanced)
        
        # Output normalization
        enhanced = self.output_norm(enhanced)
        
        return enhanced

    def score(self, x):
        """Compute scores for given quadruples"""
        # Get embeddings
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        
        # Enhance lhs embedding using MetaFormer
        enhanced = self.enhance_embeddings(lhs, rel, time)

        # lhs = lhs + enhanced 
        lhs = self.alpha * lhs + (1 - self.alpha) * enhanced
        rel = self.alpha * rel + (1 - self.alpha) * enhanced 
        
        # Split into real and imaginary parts
        lhs_r, lhs_i = lhs[:, :self.rank], lhs[:, self.rank:]
        # lhs_r, lhs_i = lhs[:, :self.rank], lhs[:, self.rank:]
        rel_r, rel_i = rel[:, :self.rank], rel[:, self.rank:]
        rhs_r, rhs_i = rhs[:, :self.rank], rhs[:, self.rank:]
        time_r, time_i = time[:, :self.rank], time[:, self.rank:]

        # TComplEx scoring
        return torch.sum(
            (lhs_r * rel_r * time_r - lhs_i * rel_i * time_r -
             lhs_i * rel_r * time_i - lhs_r * rel_i * time_i) * rhs_r +
            (lhs_i * rel_r * time_r + lhs_r * rel_i * time_r +
             lhs_r * rel_r * time_i - lhs_i * rel_i * time_i) * rhs_i,
            1, keepdim=True
        )

    def forward(self, x):
        """Forward pass for training"""
        # Get embeddings
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        
        # Enhance lhs embedding using MetaFormer
        enhanced = self.enhance_embeddings(lhs, rel, time)

        # lhs = lhs + enhanced 
        lhs = self.alpha * lhs + (1 - self.alpha) * enhanced
        rel = self.alpha * rel + (1 - self.alpha) * enhanced 
        
        # Split into real and imaginary parts
        lhs_r, lhs_i = lhs[:, :self.rank], lhs[:, self.rank:]
        # lhs_r, lhs_i = lhs[:, :self.rank], lhs[:, self.rank:]
        rel_r, rel_i = rel[:, :self.rank], rel[:, self.rank:]
        time_r, time_i = time[:, :self.rank], time[:, self.rank:]

        # Get all entity embeddings for scoring against all entities
        all_entities = self.embeddings[0].weight
        all_r, all_i = all_entities[:, :self.rank], all_entities[:, self.rank:]

        # Compute relation-time interaction
        rt_r = rel_r * time_r - rel_i * time_i
        rt_i = rel_r * time_i + rel_i * time_r

        # Compute scores against all entities
        scores = (
            (lhs_r * rt_r - lhs_i * rt_i) @ all_r.t() +
            (lhs_i * rt_r + lhs_r * rt_i) @ all_i.t()
        )

        # Return scores and regularization terms
        return scores, (
            torch.sqrt(lhs_r ** 2 + lhs_i ** 2),
            torch.sqrt(rt_r ** 2 + rt_i ** 2),
            torch.sqrt(rhs[:, :self.rank] ** 2 + rhs[:, self.rank:] ** 2)
        ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        """Forward pass over all time steps"""
        # Get embeddings
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        
        # Use average time embedding for enhancement
        avg_time = self.embeddings[2].weight.mean(0, keepdim=True).expand(lhs.size(0), -1)
        
        # Enhance lhs embedding using MetaFormer
        enhanced = self.enhance_embeddings(lhs, rel, time)

        # lhs = lhs + enhanced 
        lhs = self.alpha * lhs + (1 - self.alpha) * enhanced
        rel = self.alpha * rel + (1 - self.alpha) * enhanced 
        
        # Split into real and imaginary parts
        lhs_r, lhs_i = lhs[:, :self.rank], lhs[:, self.rank:]
        # lhs_r, lhs_i = lhs[:, :self.rank], lhs[:, self.rank:]
        rel_r, rel_i = rel[:, :self.rank], rel[:, self.rank:]
        rhs_r, rhs_i = rhs[:, :self.rank], rhs[:, self.rank:]
        
        # All time embeddings
        all_time = self.embeddings[2].weight
        time_r, time_i = all_time[:, :self.rank], all_time[:, self.rank:]

        # Compute scores over all time steps
        return (
            (lhs_r * rel_r * rhs_r - lhs_i * rel_i * rhs_r -
             lhs_i * rel_r * rhs_i + lhs_r * rel_i * rhs_i) @ time_r.t() +
            (lhs_i * rel_r * rhs_r - lhs_r * rel_i * rhs_r +
             lhs_r * rel_r * rhs_i - lhs_i * rel_i * rhs_i) @ time_i.t()
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        """Get right-hand side entities for efficient computation"""
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        """Get query representations"""
        # Get embeddings
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        
        # Enhance lhs embedding using MetaFormer
        enhanced = self.enhance_embeddings(lhs, rel, time)

        # lhs = lhs + enhanced 
        lhs = self.alpha * lhs + (1 - self.alpha) * enhanced
        rel = self.alpha * rel + (1 - self.alpha) * enhanced  
        
        # Split into real and imaginary parts
        lhs_r, lhs_i = lhs[:, :self.rank], lhs[:, self.rank:]
        # lhs_r, lhs_i = lhs[:, :self.rank], lhs[:, self.rank:]
        rel_r, rel_i = rel[:, :self.rank], rel[:, self.rank:]
        time_r, time_i = time[:, :self.rank], time[:, self.rank:]
        
        # Compute query representation
        return torch.cat([
            lhs_r * rel_r * time_r - lhs_i * rel_i * time_r -
            lhs_i * rel_r * time_i - lhs_r * rel_i * time_i,
            lhs_i * rel_r * time_r + lhs_r * rel_i * time_r +
            lhs_r * rel_r * time_i - lhs_i * rel_i * time_i
        ], 1)
