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
    pad = 0.05 # 0.1 * (global_max - global_min)

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
    fig.text(0.5, 0.65, "MLP", ha="center", fontsize=16, weight="bold")

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
            ax.set_ylabel("Prediction average", fontsize=12)

    # ----------------------- Row Title Above Row 3 -----------------------
    fig.text(0.5, 0.335, "DeepSet", ha="center", fontsize=16, weight="bold")

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
        ax.set_xlabel("Class Label")
        if i == 0:
            ax.set_ylabel("Prediction Average", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
