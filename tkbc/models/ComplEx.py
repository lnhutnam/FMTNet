from typing import Tuple, Sequence
from functools import partial, reduce
import torch
from torch import nn
import torch.nn.functional as F

from models.bases import TKBCModel


class ComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    @staticmethod
    def has_time():
        return False

    def forward_over_time(self, x):
        raise NotImplementedError("no.")

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        return (
                       (lhs[0] * rel[0] - lhs[1] * rel[1]) @ right[0].transpose(0, 1) +
                       (lhs[0] * rel[1] + lhs[1] * rel[0]) @ right[1].transpose(0, 1)
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), None

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)