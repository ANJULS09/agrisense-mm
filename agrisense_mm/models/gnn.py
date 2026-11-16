"""
AgriSense-MM: Spatial Graph Neural Network (GNN).
Models stress contagion and spatial autocorrelation across adjacent field zones.
Uses 2-layer GraphSAGE / GAT over field grid nodes.
Supports PyTorch Geometric with native PyTorch fallback for seamless execution.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def build_grid_edge_index(
    rows: int, cols: int, include_diagonals: bool = True
) -> torch.Tensor:
    """
    Constructs an edge_index [2, E] connecting adjacent zones on an R x C grid.
    Includes horizontal, vertical, and diagonal neighbors with bidirectional edges.
    """
    edges = []
    for r in range(rows):
        for c in range(cols):
            src_node = r * cols + c
            # Check 8-connected neighbors
            for dr, dc in [
                (-1, 0), (1, 0), (0, -1), (0, 1),   # 4-cardinal
                (-1, -1), (-1, 1), (1, -1), (1, 1),  # diagonals
            ]:
                if not include_diagonals and abs(dr) + abs(dc) > 1:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    dst_node = nr * cols + nc
                    edges.append((src_node, dst_node))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


# Native PyTorch SAGEConv Layer (Zero external dependency fallback)
class NativeSAGEConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, aggr: str = "mean"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Aggregate neighbor features into destination nodes
        out = torch.zeros(num_nodes, self.in_channels, device=x.device, dtype=x.dtype)
        ones = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)

        # Scatter add
        out.index_add_(0, dst, x[src])
        ones.index_add_(0, dst, torch.ones_like(dst.unsqueeze(-1), dtype=x.dtype))

        # Mean aggregation
        deg = torch.clamp(ones, min=1.0)
        out = out / deg

        return self.lin_self(x) + self.lin_neigh(out)


class SpatialZoneGNN(nn.Module):
    """
    2-Layer Graph Neural Network over Field Zones.
    Enables neighboring zones to propagate stress indicators (e.g. fungal spore spread,
    drought drying gradients, insect pest migration).
    """
    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 512,
        out_dim: int = 512,
        gnn_type: str = "sage",  # 'sage' or 'gat'
        dropout_p: float = 0.15,
        use_pyg: bool = True,
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.use_pyg = False

        if use_pyg:
            try:
                import torch_geometric.nn as pyg_nn
                if gnn_type == "gat":
                    self.conv1 = pyg_nn.GATConv(in_dim, hidden_dim // 4, heads=4, dropout=dropout_p)
                    self.conv2 = pyg_nn.GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout_p)
                else:
                    self.conv1 = pyg_nn.SAGEConv(in_dim, hidden_dim, aggr="mean")
                    self.conv2 = pyg_nn.SAGEConv(hidden_dim, out_dim, aggr="mean")
                self.use_pyg = True
            except Exception:
                self.use_pyg = False

        if not self.use_pyg:
            self.conv1 = NativeSAGEConv(in_dim, hidden_dim)
            self.conv2 = NativeSAGEConv(hidden_dim, out_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_p)

        # Residual skip connection for spatial preservation
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(
        self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [N_zones, 512]
            edge_index: Graph adjacency [2, E] (if None, identity skip is returned)

        Returns:
            x_refined: Spatially contextualized zone embeddings [N_zones, 512]
        """
        if edge_index is None or edge_index.size(1) == 0:
            return x

        residual = self.skip(x)

        # Layer 1
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = self.act(h)
        h = self.dropout(h)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = self.norm2(h)

        # Residual connection
        out = self.act(h + residual)
        return out

