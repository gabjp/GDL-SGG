import torch
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from common.MLP import PointCloudMLP
from Sets.DeepSets import DeepSet
from Sets.data import collate_pointcloud
from trainer import set_seed
from common.plot_utils import refined_plot

"""
0  → bathtub  
1  → bed  
2  → chair  
3  → desk  
4  → dresser  
5  → monitor  
6  → night_stand  
7  → sofa  
8  → table  
9  → toilet  

"""

set_seed(38)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlp_ckpt_path = "/home/gabrieljp/GDL-SGG/results/mlp-set-batchnorm/model.pth"
ds_ckpt_path  = "/home/gabrieljp/GDL-SGG/results/deep-set-batchnorm/model.pth"

transform = T.Compose([
    T.SamplePoints(1024),
    T.NormalizeScale()
])

test_dataset = ModelNet(
        root="data/ModelNet10",
        name="10",
        train=False,
        transform=transform
    )

val_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_pointcloud, shuffle=True)

ds = DeepSet(phi=[3,256,256,256], rho=[256,256,10], activation="relu").to(device)
mlp = PointCloudMLP(dim_list=[3072,4096,4096,10], activation="relu").to(device)

def load_model_weights(model, path):
    state = torch.load(path, map_location=device)
    if "state_dict" in state:   
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

load_model_weights(ds, ds_ckpt_path)
load_model_weights(mlp, mlp_ckpt_path)

print("✔ Models loaded.\n")


with torch.no_grad():
    data, labels = next(iter(val_loader))
    data, labels = data.to(device), labels.to(device)

    ds_probs = []

    mlp_probs = []

    for _ in tqdm(range(500)):
        perm = np.random.permutation(1024)
        data = data[:,perm,:]

        # DeepSet forward
        ds_outputs = ds(data)   # (B, num_classes)
        ds_prob = F.softmax(ds_outputs, dim=1)
        #ds_pred = torch.argmax(ds_probs, dim=1)
        one_hot = torch.zeros_like(ds_prob)
        one_hot[torch.arange(ds_prob.size(0)), ds_prob.argmax(dim=1)] = 1
        ds_probs.append(one_hot)

        # MLP forward (flatten point cloud)
        mlp_input = data.view(data.size(0), -1)
        mlp_outputs = mlp(mlp_input)
        mlp_prob = F.softmax(mlp_outputs, dim=1)
        #mlp_pred = torch.argmax(mlp_probs, dim=1)
        one_hot = torch.zeros_like(mlp_prob)
        one_hot[torch.arange(mlp_prob.size(0)), mlp_prob.argmax(dim=1)] = 1
        mlp_probs.append(one_hot)

ds_probs = torch.stack(ds_probs, dim=0)   # shape: (N, H, W)
mean_ds = ds_probs.mean(dim=0)       # shape: (H, W)
std_ds  = ds_probs.std(dim=0)        # shape: (H, W)

mlp_probs = torch.stack(mlp_probs, dim=0)   # shape: (N, H, W)
mean_mlp = mlp_probs.mean(dim=0)       # shape: (H, W)
std_mlp  = mlp_probs.std(dim=0)        # shape: (H, W)


#print(mean_ds.cpu().numpy())
#print(std_ds.cpu().numpy())

#print(mean_mlp.cpu().numpy())
#print(std_mlp.cpu().numpy())

print(labels.cpu().numpy())

#print(data.cpu().numpy())

refined_plot(data.cpu().numpy(), 
                                   mean_mlp.cpu().numpy(), 
                                   std_mlp.cpu().numpy(),
                                   mean_ds.cpu().numpy(), 
                                   std_ds.cpu().numpy(),
                                   labels.cpu().numpy())

plt.savefig("teste.pdf")

