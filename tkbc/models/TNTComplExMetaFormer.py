from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
import math

from models.bases import TKBCModel
from models.new_layers import FourierMixer, SimpleMetaBlock

class TNTComplExMetaFormer(TKBCModel):
    """Simplified TNTComplEx with MetaFormer enhancement"""
    
    def __init__(self, sizes: Tuple[int, int, int, int], rank: int,
                 num_blocks=2, use_fourier=True,
                 no_time_emb=False, init_size: float = 1e-2):
        super().__init__()
        self.sizes = sizes
        self.rank = rank
        self.no_time_emb = no_time_emb
        
        # TNTComplEx embeddings: entities, relations, time, no_time_relations
        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], 2 * rank, sparse=True),  # entities
            nn.Embedding(sizes[1], 2 * rank, sparse=True),  # relations
            nn.Embedding(sizes[3], 2 * rank, sparse=True),  # time
            nn.Embedding(sizes[1], 2 * rank, sparse=True)   # no_time relations
        ])
        
        # Initialize embeddings
        for emb in self.embeddings:
            emb.weight.data *= init_size
        
        # Simple embedding fusion - accommodate TNTComplEx embeddings
        # lhs (2*rank) + rel (2*rank)  + time (2*rank) + rel_no_time (2*rank) = 8*rank
        self.fusion = nn.Linear(2 * rank * 4, 2 * rank)  # Combine all embeddings
        
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

    def enhance_embeddings(self, lhs, rel, time, rel_no_time):
        """Apply MetaFormer enhancement to embeddings"""
        # Stack all embeddings [lhs: 2*rank, rel: 2*rank, time: 2*rank, rel_no_time: 2*rank]
        stacked = torch.cat([lhs, rel, time, rel_no_time], dim=-1)  # [B, 8*rank]
        
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
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        
        # Enhance lhs embedding using MetaFormer
        enhanced = self.enhance_embeddings(lhs, rel, time)

        # lhs = lhs + enhanced 
        lhs = self.alpha * lhs + (1 - self.alpha) * enhanced
        rel = self.alpha * rel + (1 - self.alpha) * enhanced 
        
        # Split into real and imaginary parts
        lhs = (lhs[:, :self.rank], lhs[:, self.rank:])
        rel = (rel[:, :self.rank], rel[:, self.rank:])
        rhs = (rhs[:, :self.rank], rhs[:, self.rank:])
        time = (time[:, :self.rank], time[:, self.rank:])
        rnt = (rel_no_time[:, :self.rank], rel_no_time[:, self.rank:])

        # TNTComplEx relation-time interaction
        rt = (rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1])
        full_rel = ((rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1])

        # TNTComplEx scoring
        return torch.sum(
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0] +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        """Forward pass for training"""
        # Get embeddings
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        
        # Enhance lhs embedding using MetaFormer
        enhanced = self.enhance_embeddings(lhs, rel, time)

        # lhs = lhs + enhanced 
        lhs = self.alpha * lhs + (1 - self.alpha) * enhanced
        rel = self.alpha * rel + (1 - self.alpha) * enhanced 
        
        # Split into real and imaginary parts
        lhs = (lhs[:, :self.rank], lhs[:, self.rank:])
        rel = (rel[:, :self.rank], rel[:, self.rank:])
        rhs = (rhs[:, :self.rank], rhs[:, self.rank:])
        time = (time[:, :self.rank], time[:, self.rank:])
        rnt = (rel_no_time[:, :self.rank], rel_no_time[:, self.rank:])

        # Get all entity embeddings for scoring against all entities
        right = self.embeddings[0].weight
        right = (right[:, :self.rank], right[:, self.rank:])

        # Compute relation-time interaction
        rt = (rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1])
        rrt = (rt[0] - rt[3], rt[1] + rt[2])
        full_rel = (rrt[0] + rnt[0], rrt[1] + rnt[1])

        # TNTComplEx scoring
        scores = (
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
        )

        # TNTComplEx regularizer with specific scaling factors
        regularizer = (
            math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
            torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
            math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

        return (scores, regularizer,
                self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight)

    def forward_over_time(self, x):
        """Forward pass over all time steps"""
        # Get embeddings
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight
        rel_no_time = self.embeddings[3](x[:, 1])

        # For time-forward, use average time for enhancement
        time_avg = time.mean(0).unsqueeze(0).repeat(lhs.shape[0], 1)
        
        # Enhance lhs embedding using MetaFormer
        enhanced = self.enhance_embeddings(lhs, rel, time)

        # lhs = lhs + enhanced 
        lhs = self.alpha * lhs + (1 - self.alpha) * enhanced
        rel = self.alpha * rel + (1 - self.alpha) * enhanced 
        
        # Split into real and imaginary parts
        lhs = (lhs[:, :self.rank], lhs[:, self.rank:])
        rel = (rel[:, :self.rank], rel[:, self.rank:])
        rhs = (rhs[:, :self.rank], rhs[:, self.rank:])
        time = (time[:, :self.rank], time[:, self.rank:])
        rnt = (rel_no_time[:, :self.rank], rel_no_time[:, self.rank:])

        # Time-dependent score
        score_time = (
            (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
             lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
            (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
             lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        
        # Time-independent base score
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        
        return score_time + base

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
        rel_no_time = self.embeddings[3](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        
        # Enhance lhs embedding using MetaFormer
        enhanced = self.enhance_embeddings(lhs, rel, time)

        # lhs = lhs + enhanced 
        lhs = self.alpha * lhs + (1 - self.alpha) * enhanced
        rel = self.alpha * rel + (1 - self.alpha) * enhanced 
        
        # Split into real and imaginary parts
        lhs = (lhs[:, :self.rank], lhs[:, self.rank:])
        rel = (rel[:, :self.rank], rel[:, self.rank:])
        time = (time[:, :self.rank], time[:, self.rank:])
        rnt = (rel_no_time[:, :self.rank], rel_no_time[:, self.rank:])

        # Compute relation-time interaction
        rt = (rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1])
        full_rel = ((rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1])

        # Return query representation
        return torch.cat([
            lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
            lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
        ], 1)