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

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

gnn.eval()
mlp.eval()

true_labels = []
gnn_predictions = []
mlp_predictions = []

with torch.no_grad():
    for batch, labels in tqdm(val_loader, desc="Evaluating"):

        batch = batch.to(device)
        labels = labels.to(device)

        # ---- GNN prediction ----
        gnn_logits = gnn(batch)
        gnn_pred = torch.argmax(gnn_logits, dim=1)

        #print(gnn_pred)

        # ---- MLP prediction ----
        mlp_logits = mlp(batch)
        mlp_pred = torch.argmax(mlp_logits, dim=1)

        #print(mlp_pred)

        # ---- store ----
        true_labels.extend(labels.cpu().tolist())
        gnn_predictions.extend(gnn_pred.cpu().tolist())
        mlp_predictions.extend(mlp_pred.cpu().tolist())

def report(name, preds):
    print(f"\n{name}:")
    print(f"Accuracy : {accuracy_score(true_labels, preds):.4f}")
    print(f"Precision: {precision_score(true_labels, preds):.4f}")
    print(f"Recall   : {recall_score(true_labels, preds):.4f}")
    print(f"F1-score : {f1_score(true_labels, preds):.4f}")


print("\n===== MUTAG VALIDATION RESULTS =====")
report("MLP Baseline", mlp_predictions)
report("GIN Model", gnn_predictions)
print("====================================\n")
