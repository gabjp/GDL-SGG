from torch_geometric.datasets import TUDataset
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import torch


def collate_mutag(batch):
    # batch: list of PyG Data objects (graphs)
    batched_graph = Batch.from_data_list(batch)   # merges graphs into a single Batch
    y = torch.cat([data.y for data in batch], dim=0)  # -> [batch_size]
    return batched_graph, y


def get_MUTAG_dataloader(batch_size=32, use_node_attr=True):

    dataset = TUDataset(
        root="data/MUTAG",
        name="MUTAG",
        use_node_attr=use_node_attr
    )

    # Standard split (80/20)
    num_train = int(0.8 * len(dataset))
    num_test = len(dataset) - num_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_mutag
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_mutag, 
        shuffle=True
    )

    return train_loader, test_loader


