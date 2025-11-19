from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
import torch

def collate_pointcloud(batch):
    # batch: list of Data objects (PyG)
    xs = [data.pos for data in batch]          # list of [num_points, 3]
    ys = [data.y for data in batch]            # list of [1]

    x = torch.stack(xs, dim=0)                 # -> [batch_size, num_points, 3]
    y = torch.cat(ys, dim=0)                   # -> [batch_size]

    return x, y


def get_ModelNet_dataloader(num_points = 1024, batch_size=8):

    transform = T.Compose([
        T.SamplePoints(num_points),
        T.NormalizeScale()
    ])

    train_dataset = ModelNet(
        root="data/ModelNet10",
        name="10",
        train=True,
        transform=transform
    )

    test_dataset = ModelNet(
        root="data/ModelNet10",
        name="10",
        train=False,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pointcloud)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_pointcloud)

    return train_loader, test_loader

