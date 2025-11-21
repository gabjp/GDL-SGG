import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from common.MLP import GraphMLP
from Graphs.GNN import GIN

from trainer import set_seed
from common.plot_utils import refined_plot, refined_plot_graphs

from Graphs.data import get_MUTAG_dataloader

"""
0  → not mutagenic  
1  → mutagenic  
"""

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gnn_ckpt_path = "/home/gabrieljp/GDL-SGG/results/GNN-batchnorm/model.pth"
mlp_ckpt_path  = "/home/gabrieljp/GDL-SGG/results/MLPGraphs-batchnorm/model.pth"


_,val_loader = get_MUTAG_dataloader(batch_size=1)

gnn = GIN(in_dim = 7, hidden_dim = 64, mlp = [], num_classes = 2).to(device)
mlp = GraphMLP(dim_list=[980, 64, 64, 2], activation="relu").to(device)

def load_model_weights(model, path):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()

load_model_weights(gnn, gnn_ckpt_path)
load_model_weights(mlp, mlp_ckpt_path)

print("✔ Models loaded.\n")

gnn_probs = []

mlp_probs = []

labelsss = []

indexes = []

with torch.no_grad():
    for i in range(25):
        data, labels = next(iter(val_loader))
        data, labels = data.to(device), labels.to(device)

        labelsss.append(labels.cpu().item())

        x, edge_index, batch = data.x, data.edge_index, data.batch
        indexes.append(edge_index)
        #print(x.shape)
        #print(edge_index.shape)
        #print(batch.shape)

        gnn_preds = []
        mlp_preds = []

        for _ in tqdm(range(500)):
            perm = torch.randperm(len(batch)).cuda()
            data.x = x[perm]

            inv_perm = torch.empty_like(perm).cuda()
            inv_perm[perm] = torch.arange(len(perm), device=perm.device)

            data.edge_index = inv_perm[edge_index]

            # DeepSet forward
            gnn_outputs = gnn(data).squeeze()  # (B, num_classes)
            mlp_outputs = mlp(data).squeeze()

            gnn_onehot = torch.zeros_like(gnn_outputs)
            gnn_onehot[torch.argmax(gnn_outputs)] = 1

            mlp_onehot = torch.zeros_like(mlp_outputs)
            mlp_onehot[torch.argmax(mlp_outputs)] = 1

            # MLP forward (flatten point cloud)
            gnn_preds.append(gnn_onehot)
            mlp_preds.append(mlp_onehot)
        
        gnn_probs.append(torch.stack(gnn_preds, dim=0).mean(dim=0))
        mlp_probs.append(torch.stack(mlp_preds, dim=0).mean(dim=0))

gnn_probs = torch.stack(gnn_probs, dim=0)
mlp_probs = torch.stack(mlp_probs, dim=0)
labelsss = torch.tensor(labelsss)
indexes = [indexes[i] for i in [-1, -2, -3, -5]]

#print(gnn_probs[[-1, -2, -3, -5]])
#print(mlp_probs[[-1, -2, -3, -5]])
#print(labelsss[[-1, -2, -3, -5]])
#print([indexes[i] for i in [-1, -2, -3, -5]])

fig = refined_plot_graphs(
    edge_index_list=indexes,
    mlp_preds=mlp_probs[[-1, -2, -3, -5]],
    gnn_preds=gnn_probs[[-1, -2, -3, -5]],
    labels=labelsss[[-1, -2, -3, -5]]
)
fig.savefig("teste.pdf")