from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import scipy.io as scio

from torch_geometric.utils import coalesce, scatter, softmax


class EdgePooling(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_score_method: Optional[Callable] = None,
        dropout: Optional[float] = 0.0,
        add_to_edge_score: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_sigmoid
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout

        self.lin = torch.nn.Linear(2 * in_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(
        raw_edge_score: Tensor,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        r"""Normalizes edge scores via softmax application."""
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_tanh(
        raw_edge_score: Tensor,
        edge_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Normalizes edge scores via hyperbolic tangent application."""
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(
        raw_edge_score: Tensor,
        edge_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Normalizes edge scores via sigmoid application."""
        return torch.sigmoid(raw_edge_score)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The node features.
            edge_index (torch.Tensor): The edge indices.
            batch (torch.Tensor): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.

        Return types:
            * **x** *(torch.Tensor)* - The pooled node features.
            * **edge_index** *(torch.Tensor)* - The coarsened edge indices.
            * **batch** *(torch.Tensor)* - The coarsened batch vector.
            * **unpool_info** *(UnpoolInfo)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """

        edge_score = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_score = F.relu(self.lin(edge_score))
        edge_score = F.dropout(edge_score, p=self.dropout, training=self.training)
        edge_score = self.compute_edge_score(edge_score, edge_index, x.size(0))
        new_edge_index1, new_edge_score1 = self.edge_delete_adap(batch, edge_score)
        new_edge_index2, new_edge_score2 = self.edge_delete_adap(batch, edge_score)
        
        return new_edge_index1, new_edge_index2, batch, new_edge_score


    def edge_delete_adap(
            self,
            batch: Tensor,
            edge_score: Tensor,
    ):
        edges = len(edge_score) / (max(batch) + 1)
        nodes = int(torch.sqrt(edges))
        edges = torch.tensor(edges.ceil(), dtype=torch.long)
        edge_score = torch.reshape(edge_score, [max(batch) + 1, edges])
        min_val, _ = torch.min(edge_score, dim=1, keepdim=True)
        max_val, _ = torch.max(edge_score, dim=1, keepdim=True)
        edge_score = (edge_score - min_val) / (max_val - min_val + 1e-6)
        rate = torch.rand_like(edge_score)
        perm = torch.where(rate > (1 - edge_score), torch.tensor(1).cuda(), torch.tensor(0).cuda())
        pos = torch.nonzero(perm)
        new_edge_index0 = torch.div(pos[:, 1], nodes, rounding_mode='trunc') + pos[:, 0] * nodes
        new_edge_index1 = torch.remainder(pos[:, 1], nodes) + pos[:, 0] * nodes
        new_edge_index = torch.concat((new_edge_index0.unsqueeze(0), new_edge_index1.unsqueeze(0)), dim=0)

        return new_edge_index, edge_score

    def edge_delete(
        self,
        edge_index: Tensor,
        batch: Tensor,
        edge_score: Tensor,
        ratio: torch.float
    ):
        edges = len(edge_score) / (max(batch) + 1)
        edges = torch.tensor(edges.ceil(), dtype=torch.long)
        edge_score = torch.reshape(edge_score, [max(batch)+1, edges])
        perm = torch.argsort(edge_score, descending=True, dim=1)
        left_edges = int((max(batch) + 1) * edges * ratio)
        left_edges_pg = int(edges * ratio)
        new_edge_index = torch.zeros([2, left_edges], dtype=torch.long)
        for i in range(len(perm)):
            pos = i * edges + perm[i, 0:left_edges_pg]
            new_edge_index[0, i*left_edges_pg:(i+1)*left_edges_pg] = edge_index[0, pos]
            new_edge_index[1, i * left_edges_pg:(i + 1) * left_edges_pg] = edge_index[1, pos]

        return new_edge_index
