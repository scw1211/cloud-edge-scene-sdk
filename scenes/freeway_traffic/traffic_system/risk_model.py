"""用途：定义联合 ASTGCN 的节点风险头、区域风险头和区域聚合逻辑。"""

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ASTGCN_r import make_model


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointRiskASTGCN(nn.Module):
    """ASTGCN forecast backbone with node-level and region-level risk heads."""

    def __init__(
        self,
        device: torch.device,
        nb_block: int,
        in_channels: int,
        k_order: int,
        nb_chev_filter: int,
        nb_time_filter: int,
        time_strides: int,
        adj_mx: np.ndarray,
        num_for_predict: int,
        len_input: int,
        num_of_vertices: int,
        partitions: Sequence[Sequence[int]],
        output_dim: int = 3,
        num_risk_classes: int = 4,
        risk_hidden_dim: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = make_model(
            device,
            nb_block,
            in_channels,
            k_order,
            nb_chev_filter,
            nb_time_filter,
            time_strides,
            adj_mx,
            num_for_predict,
            len_input,
            num_of_vertices,
            output_dim=output_dim,
        )
        self.node_head = MLPHead(nb_time_filter, risk_hidden_dim, num_risk_classes, dropout)
        self.region_head = MLPHead(nb_time_filter, risk_hidden_dim, num_risk_classes, dropout)

        mask = torch.zeros(len(partitions), num_of_vertices, dtype=torch.float32)
        for region_idx, node_ids in enumerate(partitions):
            mask[region_idx, [int(node_id) for node_id in node_ids]] = 1.0
        self.register_buffer("region_mask", mask)
        self.register_buffer("region_size", mask.sum(dim=1).clamp_min(1.0))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.backbone.encode(x)
        forecast = self.backbone.decode_forecast(encoded)

        node_repr = encoded.mean(dim=-1)
        node_logits = self.node_head(node_repr)

        region_repr = torch.einsum("rn,bnc->brc", self.region_mask, node_repr)
        region_repr = region_repr / self.region_size.view(1, -1, 1)
        region_logits = self.region_head(region_repr)
        return {
            "forecast": forecast,
            "node_logits": node_logits,
            "region_logits": region_logits,
            "node_repr": node_repr,
            "region_repr": region_repr,
        }


def aggregate_node_probs_to_regions(
    node_logits: torch.Tensor,
    region_mask: torch.Tensor,
    region_size: torch.Tensor,
) -> torch.Tensor:
    node_probs = F.softmax(node_logits, dim=-1)
    region_probs = torch.einsum("rn,bnk->brk", region_mask, node_probs)
    return region_probs / region_size.view(1, -1, 1)
