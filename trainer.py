import argparse
import torch
import numpy as np
import random
import yaml
import os
import json
from torch.optim.lr_scheduler import ExponentialLR

from common.MLP import PointCloudMLP, GraphMLP, ImageMLP
from common.train_utils import train_model, linear_scheduler
from Sets.DeepSets import DeepSet
from Sets.data import get_ModelNet_dataloader
from Graphs.data import get_MUTAG_dataloader
from Graphs.GNN import GIN
from Grids.data import get_MNIST_dataloader
from Grids.CNN import CNN

def set_seed(seed: int = 42):
    print(f"Setting seed: {seed}")

    # Python built-ins
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU and GPU seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Ensure deterministic behavior when possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():

    parser = argparse.ArgumentParser(description="Training script for model")

    parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="Dataset name",
            choices = [ "ModelNet10", "MUTAG", "MNIST"]
        )

    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0, help="Weight Decay")

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd", "adamw"],
        help="Optimizer type"
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "log", "linear"],
        help="LR scheduler"
    )

    # ---------------- Model config ----------------
    parser.add_argument(
        "--model-type", 
        type=str, 
        default="MLP", 
        help="Model type",
        choices = ["MLP", "DeepSet", "CNN", "GNN"]
    )
    
    parser.add_argument("--config-path", type=str, help="Model_config_path")

    # ---------------- System/runtime flags ----------------
    parser.add_argument("--device", type=str, default="cuda", help="Device: cpu or cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exp_name", type=str, help="Experiment Name")

    # Parse args
    args = parser.parse_args()

    print("\n===== Args =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("==================\n")

    return args

def main():

    # Load Args
    args = get_args()
    # Set seed
    set_seed(args.seed)

    #Load config YAML
    with open(args.config_path, "r") as f:
        model_kwargs = yaml.safe_load(f)

    criterion = torch.nn.CrossEntropyLoss()

    if args.dataset == "ModelNet10":
        train_loader, val_loader = get_ModelNet_dataloader(batch_size=args.batch_size)
        if args.model_type == "MLP":
            model = PointCloudMLP(**model_kwargs)
        elif args.model_type == "DeepSet":
            model = DeepSet(**model_kwargs)

        
    elif args.dataset == "MUTAG":
        train_loader, val_loader = get_MUTAG_dataloader(batch_size=args.batch_size)
        if args.model_type == "MLP":
            model = GraphMLP(**model_kwargs)
        elif args.model_type == "GNN":
            model = GIN(**model_kwargs)

    elif args.dataset == "MNIST":
        train_loader, val_loader = get_MNIST_dataloader(batch_size=args.batch_size)
        if args.model_type == "MLP":
            model = ImageMLP(**model_kwargs)
        elif args.model_type == "CNN":
            model = CNN(**model_kwargs)


    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", n_params)

    assert False
    
    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        opt =  torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)  
    elif args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "log":
        scheduler = ExponentialLR(opt, gamma=0.95)
    elif args.scheduler == "linear":
        scheduler = linear_scheduler(opt, total_epochs = args.epochs)
    elif args.scheduler == "none":
        scheduler = None
    else: 
        raise NotImplementedError()
    
    results = train_model(
        model = model,
        train_loader=train_loader,
        val_loader=val_loader, 
        optimizer=opt,
        criterion = criterion,
        scheduler = scheduler,
        device = args.device,
        epochs = args.epochs
    )

    directory = f"results/{args.exp_name}"

    # Create directories if needed
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    # Save the model weights (.pth)
    model_save_path = directory + "/model.pth"
    torch.save(model.state_dict(), model_save_path)

    # Save the config dictionary as JSON
    json_path = directory + "/results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {json_path}")



if __name__ == "__main__":
    main()