import torch
import torch.nn as nn
from typing import Literal
from torch_geometric.utils import to_dense_adj

def batch_to_fixed_vectors(x, edge_index, batch):
    MAX_NODES = 28
    feature_dim = x.size(1)

    # number of graphs in batch
    num_graphs = int(batch.max().item() + 1)

    # offsets: where each graph starts
    # Example: batch = [0,0,0,1,1,2,2,2] -> idx_per_graph = [0,3,5,8]
    idx_per_graph = torch.cumsum(torch.bincount(batch), dim=0)
    idx_per_graph = torch.cat([torch.tensor([0], device=batch.device), idx_per_graph])

    vectors = []

    for g in range(num_graphs):
        start, end = idx_per_graph[g].item(), idx_per_graph[g+1].item()

        # slice nodes
        x_g = x[start:end]

        # remap edge_index so that graph-local node numbering starts at 0
        mask = (edge_index[0] >= start) & (edge_index[0] < end)
        ei = edge_index[:, mask] - start

        # ---- pad node features ----
        x_pad = torch.zeros((MAX_NODES, feature_dim), device=x.device)
        x_pad[:x_g.size(0)] = x_g

        # ---- adjacency ----
        A = to_dense_adj(ei, max_num_nodes=MAX_NODES)[0]

        # ---- flatten ----
        vec = torch.cat([A.reshape(-1), x_pad.reshape(-1)], dim=0)
        #print(sum(vec))
        vectors.append(vec)

    return torch.stack(vectors)
  # -> [batch_size, vector_dim]


class MLP(nn.Module):

    def __init__(
        self,
        dim_list: list[int] = [10,32,10],
        activation: Literal["relu", "gelu", "tanh", "leaky_relu"] = "relu"
    ):
        super().__init__()

        num_layers = len(dim_list) - 1

        assert num_layers >= 1, "number of layers must be >= 1"

        # ----- Select activation -----
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        act = activations[activation]

        layers = []

        for i in range(num_layers):
            layers.append(nn.Linear(dim_list[i], dim_list[i+1]))

            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(dim_list[i+1]))  
                layers.append(act)

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PointCloudMLP(MLP):
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.model(x)
    
class GraphMLP(MLP):
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch # max is 28 nodes
        x = batch_to_fixed_vectors(x, edge_index, batch)
        return self.model(x)
        
