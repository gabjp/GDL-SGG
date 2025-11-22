import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from common.MLP import ImageMLP
from Grids.CNN import CNN

from trainer import set_seed
from common.plot_utils import refined_plot_mnist

from Grids.data import get_MNIST_dataloader

"""
0  → not mutagenic  
1  → mutagenic  
"""

def translate(img):

    H = W = 28   # MNIST size

    # ---- random shifts ----
    dx = torch.randint(0, W, ()).item()   # horizontal
    dy = torch.randint(0, H, ()).item()   # vertical

    #print(f"Generated shifts → dx={dx}, dy={dy}")

    # ---- build permutation matrix for x-shift ----
    Px = torch.zeros(W, W)
    for i in range(W):
        Px[i, (i - dx) % W] = 1   # circular shift

    # ---- build permutation matrix for y-shift ----
    Py = torch.zeros(H, H)
    for j in range(H):
        Py[j, (j - dy) % H] = 1

    # ---- 2D translation operator (Kronecker product) ----
    T = torch.kron(Py, Px)   # shape (784×784)


    shifted = (T.cuda() @ img.flatten()).reshape(1, 1, H, W)

    return shifted


set_seed(34)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_ckpt_path = "/home/gabrieljp/GDL-SGG/results/CNN-batchnorm/model.pth"
mlp_ckpt_path  = "/home/gabrieljp/GDL-SGG/results/MLPGrids-batchnorm/model.pth"


_,val_loader = get_MNIST_dataloader(batch_size=1)

cnn = CNN(conv1 =16, conv2 = 16, conv3 = 32, conv4 = 32, num_classes = 10).to(device)
mlp = ImageMLP(dim_list=[784, 64, 64, 10], activation="relu").to(device)

def load_model_weights(model, path):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()

load_model_weights(cnn, cnn_ckpt_path)
load_model_weights(mlp, mlp_ckpt_path)

print("✔ Models loaded.\n")

cnn_probs = []

mlp_probs = []

labelsss = []

imgs = []

with torch.no_grad():
    for i in range(4):
        data, labels = next(iter(val_loader))
        data, labels = data.to(device), labels.to(device)

        labelsss.append(labels.cpu().item())
        imgs.append(data.cpu()[0][0])

        cnn_preds = []
        mlp_preds = []

        for _ in tqdm(range(500)):
            data = data[0]
            data = translate(data)

            # DeepSet forward
            cnn_outputs = cnn(data)[0] # (B, num_classes)
            mlp_outputs = mlp(data)[0]

            cnn_onehot = torch.zeros_like(cnn_outputs)
            cnn_onehot[torch.argmax(cnn_outputs)] = 1

            mlp_onehot = torch.zeros_like(mlp_outputs)
            mlp_onehot[torch.argmax(mlp_outputs)] = 1

            # MLP forward (flatten point cloud)
            cnn_preds.append(cnn_onehot)
            mlp_preds.append(mlp_onehot)
        
        cnn_probs.append(torch.stack(cnn_preds, dim=0).mean(dim=0))
        mlp_probs.append(torch.stack(mlp_preds, dim=0).mean(dim=0))

cnn_probs = torch.stack(cnn_probs, dim=0)
mlp_probs = torch.stack(mlp_probs, dim=0)
labelsss = torch.tensor(labelsss)

print(cnn_probs)
print(mlp_probs)
print(labelsss)
#print(imgs)

fig = refined_plot_mnist(imgs, mlp_probs, cnn_probs, labelsss)
plt.savefig("teste.pdf")