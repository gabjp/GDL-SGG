import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader


def get_MNIST_dataloader(batch_size=32):

    transform = transforms.Compose([
        transforms.ToTensor(),       # convert to [0,1] tensor
        transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization (optional)
    ])

    train_dataset = MNIST(
        root="data/MNIST",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = MNIST(
        root="data/MNIST",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, test_loader
