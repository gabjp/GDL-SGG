import torch
import torch.nn as nn
from typing import Literal

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
        
