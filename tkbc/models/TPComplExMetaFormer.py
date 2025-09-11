from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F

from models.bases import TKBCModel
from models.new_layers import FourierMixer, SimpleMetaBlock

class TPComplExMetaFormer(TKBCModel):
    """Simplified TPComplEx with MetaFormer enhancement"""
    
    def __init__(self, sizes: Tuple[int, int, int, int], rank: int,
                 num_blocks=1, use_fourier=True,
                 no_time_emb=False, init_size: float = 1e-2):
        super().__init__()
        self.sizes = sizes
        self.rank = rank
        self.no_time_emb = no_time_emb
        
        # TPComplEx embeddings: entities (2*rank), relations (2*rank), time (6*rank)
        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], 2 * rank, sparse=True),  # entities
            nn.Embedding(sizes[1], 2 * rank, sparse=True),  # relations
            nn.Embedding(sizes[3], 6 * rank, sparse=True)   # time (6*rank for TPComplEx)
        ])
        
        # Initialize embeddings
        for emb in self.embeddings:
            emb.weight.data *= init_size
        
        # Simple embedding fusion - accommodate TPComplEx's larger time embedding
        # lhs (2*rank) + rel (2*rank) + time (6*rank) = 10*rank
        self.fusion = nn.Linear(2 * rank * 5, 2 * rank)  # Combine all embeddings
        
        # MetaFormer blocks
        self.meta_blocks = nn.ModuleList([
            SimpleMetaBlock(2 * rank, use_fourier) 
            for _ in range(num_blocks)
        ])
        
        # Output normalization
        self.output_norm = nn.LayerNorm(2 * rank)

        self.alpha = 0.85

    @staticmethod
    def has_time():
        return True

    def enhance_embeddings(self, lhs, rel, time):
        """Apply MetaFormer enhancement to embeddings"""
        # Stack all embeddings [lhs: 2*rank, rel: 2*rank, time: 6*rank]
        stacked = torch.cat([lhs, rel, time], dim=-1)  # [B, 12*rank]
        
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
        
        # TPComplEx time bias adjustments
        lhs = (lhs[:, :self.rank] + time[:, 2*self.rank:3*self.rank], 
               lhs[:, self.rank:] + time[:, 3*self.rank:4*self.rank])
        rhs = (rhs[:, :self.rank] + time[:, 4*self.rank:5*self.rank], 
               rhs[:, self.rank:] + time[:, 5*self.rank:6*self.rank])
        rel = (rel[:, :self.rank], rel[:, self.rank:2*self.rank])
        time = (time[:, :self.rank], time[:, self.rank:2*self.rank])

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]), (rt[1] + rt[2])

        return torch.sum(
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0]
            + (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1],
            1,
            keepdim=True,
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
        
        # TPComplEx time bias adjustments
        lhs = (lhs[:, :self.rank] + time[:, 2*self.rank:3*self.rank], 
               lhs[:, self.rank:] + time[:, 3*self.rank:4*self.rank])
        rhs = (rhs[:, :self.rank] + time[:, 4*self.rank:5*self.rank], 
               rhs[:, self.rank:] + time[:, 5*self.rank:6*self.rank])
        
        # Bias terms from time embedding
        bias_t_r = time[:, 4*self.rank:5*self.rank]
        bias_t_i = time[:, 5*self.rank:6*self.rank]
        
        time = (time[:, :self.rank], time[:, self.rank:2*self.rank])
        rel = (rel[:, :self.rank], rel[:, self.rank:2*self.rank])

        # Get all entity embeddings for scoring against all entities
        right = self.embeddings[0].weight
        right = (right[:, :self.rank], right[:, self.rank:])

        # Compute relation-time interaction
        rt = (rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1])
        full_rel = (rt[0] - rt[3], rt[1] + rt[2])

        return (
            (
                (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t()
                + torch.sum(
                    (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * bias_t_r,
                    1,
                    keepdim=True,
                )
                + (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
                + torch.sum(
                    (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * bias_t_i,
                    1,
                    keepdim=True,
                )
            ),
            (
                torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
            ),
            (
                self.embeddings[2].weight[:-1]
                if self.no_time_emb
                else self.embeddings[2].weight
            ),
        )

    def forward_over_time(self, x):
        """Forward pass over all time steps - not implemented for TPComplEx"""
        raise NotImplementedError("forward_over_time not implemented for TPComplEx")

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
        
        # TPComplEx time bias adjustments
        lhs = (lhs[:, :self.rank] + time[:, 2*self.rank:3*self.rank], 
               lhs[:, self.rank:] + time[:, 3*self.rank:4*self.rank])
        rel = (rel[:, :self.rank], rel[:, self.rank:2*self.rank])
        time = (time[:, :self.rank], time[:, self.rank:2*self.rank])

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]), (rt[1] + rt[2])
        
        return torch.cat(
            [
                lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
                lhs[1] * full_rel[0] + lhs[0] * full_rel[1],
            ],
            1,
        )