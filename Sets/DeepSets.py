
import torch
import torch.nn as nn
from common.MLP import MLP

class DeepSet(nn.Module):
    def __init__(self, phi = [3, 256, 256, 256], rho = [256, 256, 10], activation="relu"):
        super().__init__()
        self.phi = MLP(dim_list=phi, activation=activation)
        self.rho = MLP(dim_list=rho, activation=activation)

    def forward(self, x):
        # compute the representation for each data point
        #(B, 1024, 3)
        B, N, D = x.shape
        x = x.reshape(B * N, D)
        
        x = self.phi.forward(x)
        x = x.reshape(B,N, -1)
        

        x = torch.mean(x, dim=-2, keepdim=False) # mean aggregation
        # compute the output
        out = self.rho.forward(x)

        return out
    
