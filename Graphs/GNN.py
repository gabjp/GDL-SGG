import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool

from common.MLP import MLP


class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, mlp, num_classes):
        super().__init__()

        # GIN uses an MLP inside the conv layer
        
        self.conv1 = GINConv(MLP([in_dim, hidden_dim, hidden_dim]))
        self.conv2 = GINConv(MLP([hidden_dim, hidden_dim, hidden_dim]))
        self.conv3 = GINConv(MLP([hidden_dim, hidden_dim, hidden_dim]))

        # classifier same as before
        self.classifier = MLP([hidden_dim] + mlp + [num_classes])
        #self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # ===== Layer 1 =====
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)

        # ===== Graph-level pooling =====
        x = global_add_pool(x, batch)

        return self.classifier(x)