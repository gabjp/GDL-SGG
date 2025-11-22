import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def refined_plot(
    point_clouds,
    means_A, stds_A,
    means_B, stds_B,
    labels,
    figsize=(20, 11)  # slightly larger
):

    # ---- Convert to numpy (allows torch tensors) ----
    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    point_clouds = to_numpy(point_clouds)
    means_A, stds_A = to_numpy(means_A), to_numpy(stds_A)
    means_B, stds_B = to_numpy(means_B), to_numpy(stds_B)

    # ---- Validate shapes ----
    assert point_clouds.shape == (4, 1024, 3)
    assert means_A.shape == stds_A.shape == (4, 10)
    assert means_B.shape == stds_B.shape == (4, 10)

    num_clouds = 4
    x_idxs = np.arange(10)

    # ---- ModelNet10 class names ----
    modelnet10_labels = [
        "bathtub", "bed", "chair", "desk", "dresser",
        "monitor", "night_stand", "sofa", "table", "toilet"
    ]

    # ---- Colors ----
    color_A = "#C65A5A"
    color_B = "#8E3B3B"
    point_color = "#CC4C4C"

    # ---- Global histogram scaling ----
    global_min = 0#min(means_A.min(), means_B.min())
    global_max = 1#max((means_A + stds_A).max(), (means_B + stds_B).max())
    pad = 0.0 # 0.1 * (global_max - global_min)

    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    fig = plt.figure(figsize=figsize)
    #fig.text(0.5, 0.97, "Prediction Count per Permutation", ha="center", fontsize=16, weight="bold")


    # ----------------------- Row 1: Point Clouds -----------------------
    for i in range(num_clouds):
        ax = fig.add_subplot(3, 4, i + 1, projection="3d")
        pc = point_clouds[i]

        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=8, alpha=0.7, c=point_color)

        ax.set_title(f"Label: {modelnet10_labels[labels[i]]}",  fontsize=16, weight="bold")

        # ---- Remove ticks and numbers while keeping axes ----
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

        # Optional: remove tick marks also
        ax.xaxis.set_tick_params(size=0)
        ax.yaxis.set_tick_params(size=0)
        ax.zaxis.set_tick_params(size=0)

    # ----------------------- Row Title Above Row 2 -----------------------
    fig.text(0.5, 0.65, "MLP Predictions", ha="center", fontsize=16, weight="bold")

    # ----------------------- Row 2: Histograms A -----------------------
    for i in range(num_clouds):
        ax = fig.add_subplot(3, 4, 4 + i + 1)

        ax.set_facecolor("#F2F2F2")

        ax.bar(x_idxs, means_A[i],
               capsize=3, alpha=0.85, color=color_A)

        ax.set_ylim(global_min - pad, global_max + pad)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.18)

        ax.set_xticks(x_idxs)
        ax.set_xticklabels(modelnet10_labels, rotation=45, ha="right", fontsize=8)

        if i == 0:
            ax.set_ylabel("Prediction Average", fontsize=12)

    # ----------------------- Row Title Above Row 3 -----------------------
    fig.text(0.5, 0.335, "DeepSet Predictions", ha="center", fontsize=16, weight="bold")

    # ----------------------- Row 3: Histograms B -----------------------
    for i in range(num_clouds):
        ax = fig.add_subplot(3, 4, 8 + i + 1)

        ax.set_facecolor("#F2F2F2")
        ax.bar(x_idxs, means_B[i],
               capsize=3, alpha=0.85, color=color_B)

        ax.set_ylim(global_min - pad, global_max + pad)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.18)

        ax.set_xticks(x_idxs)
        ax.set_xticklabels(modelnet10_labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Class")
        if i == 0:
            ax.set_ylabel("Prediction Average", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch


def refined_plot_graphs(
    edge_index_list,      # list of 4 PyG edge_index tensors
    mlp_preds,            # shape: (4, 2)
    gnn_preds,            # shape: (4, 2)
    labels,              # shape: (4,)
    figsize=(20, 11)
):

    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    
    edge_index_list = [to_numpy(ei) for ei in edge_index_list]
    mlp_preds = to_numpy(mlp_preds)
    gnn_preds = to_numpy(gnn_preds)
    labels = to_numpy(labels)

    assert len(edge_index_list) == 4
    assert mlp_preds.shape == gnn_preds.shape == (4, 2)

    num_graphs = 4
    x_idxs = np.arange(2)

    # ---- MUTAG labels ----
    mutag_labels = ["Non-Mutagenic", "Mutagenic"]

    # ---- Color Palette ----
    color_A = "#C65A5A"   # MLP
    color_B = "#8E3B3B"   # GNN
    node_color = "#CC4C4C"
    background_color = "#F6EAEA"

    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    fig = plt.figure(figsize=figsize)

    # ----------------------- Row 1: Graphs -----------------------
    for i in range(num_graphs):
        ax = fig.add_subplot(3, 4, i + 1)

        edges = list(zip(edge_index_list[i][0], edge_index_list[i][1]))
        G = nx.Graph()
        G.add_edges_from(edges)

        pos = nx.spring_layout(G, seed=42)

        nx.draw(
            G, pos, ax=ax,
            node_size=350,
            node_color=node_color,
            edge_color=color_B,
            width=2,
            labels={},                # <--- removes node numbers
        )

        ax.set_title(
            f"Label: {mutag_labels[int(labels[i])]}", 
            fontsize=14, weight="bold"
        )

    # ----------------------- Title Row 2 -----------------------
    fig.text(0.5, 0.65, "MLP Predictions", ha="center", fontsize=16, weight="bold")

    # ----------------------- Row 2: MLP Histograms -----------------------
    for i in range(num_graphs):
        ax = fig.add_subplot(3, 4, 4 + i + 1)
        ax.set_facecolor(background_color)

        ax.bar(x_idxs, mlp_preds[i], capsize=3, alpha=0.9, color=color_A)
        ax.set_facecolor("#F2F2F2")
        ax.set_ylim(0, 1)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.18,)

        ax.set_xticks(x_idxs)
        ax.set_xticklabels(mutag_labels, rotation=15, fontsize=10)
        if i == 0:
            ax.set_ylabel("Prediction Average", fontsize=12)

    # ----------------------- Title Row 3 -----------------------
    fig.text(0.5, 0.335, "GNN Predictions", ha="center", fontsize=16, weight="bold")

    # ----------------------- Row 3: GNN Histograms -----------------------
    for i in range(num_graphs):
        ax = fig.add_subplot(3, 4, 8 + i + 1)
        ax.set_facecolor(background_color)

        ax.bar(x_idxs, gnn_preds[i], capsize=3, alpha=0.9, color=color_B)
        ax.set_facecolor("#F2F2F2")
        ax.set_ylim(0, 1)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.18)

        ax.set_xticks(x_idxs)
        ax.set_xticklabels(mutag_labels, rotation=15, fontsize=10)
        ax.set_xlabel("Class")
        if i == 0:
            ax.set_ylabel("Prediction Average", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


import matplotlib.pyplot as plt
import numpy as np

def refined_plot_mnist(
    images,          # list of 4 MNIST tensors [28, 28] or [1, 28, 28]
    mlp_preds,       # shape: (4, N_CLASSES)
    cnn_preds,       # shape: (4, N_CLASSES)
    labels,          # shape: (4,)
    figsize=(20, 11)
):

    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # ---- Convert inputs ----
    images = [to_numpy(img).squeeze() for img in images]
    mlp_preds = to_numpy(mlp_preds)
    cnn_preds = to_numpy(cnn_preds)
    labels = to_numpy(labels)

    assert len(images) == 4
    assert mlp_preds.shape == cnn_preds.shape
    assert mlp_preds.shape[0] == 4

    num_items = 4
    num_classes = mlp_preds.shape[1]
    x_idxs = np.arange(num_classes)
    class_names = [str(i) for i in range(num_classes)]

    # ---- Colors ----
    color_A = "#C65A5A"      # MLP bars
    color_B = "#8E3B3B"      # CNN bars
    hist_bg = "#F2F2F2"      # GRAY histogram background

    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    fig = plt.figure(figsize=figsize)

    # ----------------------- Row 1: MNIST Images -----------------------
    for i in range(num_items):
        ax = fig.add_subplot(3, 4, i + 1)
        ax.set_facecolor("white")   # image background WHITE

        img = images[i]
        # invert: background white, digit black
        img_disp = img.max() - img

        ax.imshow(img_disp, cmap="gray")
        ax.axis("off")
        ax.set_title(
            f"Label: {class_names[int(labels[i])]}",
            fontsize=14, weight="bold"
        )

    # ----------------------- Title Row 2 -----------------------
    fig.text(0.5, 0.64, "MLP Predictions", ha="center",
             fontsize=14, weight="bold")

    # ----------------------- Row 2: MLP Histograms -----------------------
    for i in range(num_items):
        ax = fig.add_subplot(3, 4, 4 + i + 1)
        ax.set_facecolor(hist_bg)   # HISTOGRAM BACKGROUND GRAY

        ax.bar(x_idxs, mlp_preds[i], capsize=3, alpha=0.9, color=color_A)
        ax.set_ylim(0, 1)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.18)

        ax.set_xticks(x_idxs)
        ax.set_xticklabels(class_names, rotation=15, fontsize=10)
        if i == 0:
            ax.set_ylabel("Prediction Average", fontsize=12)  # <- fixed

    # ----------------------- Title Row 3 -----------------------
    fig.text(0.5, 0.327, "CNN Predictions", ha="center",
             fontsize=14, weight="bold")

    # ----------------------- Row 3: CNN Histograms -----------------------
    for i in range(num_items):
        ax = fig.add_subplot(3, 4, 8 + i + 1)
        ax.set_facecolor(hist_bg)   # HISTOGRAM BACKGROUND GRAY

        ax.bar(x_idxs, cnn_preds[i], capsize=3, alpha=0.9, color=color_B)
        ax.set_ylim(0, 1)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.18)

        ax.set_xticks(x_idxs)
        ax.set_xticklabels(class_names, rotation=15, fontsize=10)
        ax.set_xlabel("Class")
        if i == 0:
            ax.set_ylabel("Prediction Average", fontsize=12)  # <- fixed

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
