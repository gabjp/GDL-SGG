import torch
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[Callable] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[str] = None,
    epochs: int = 20,
) -> Dict[str, list]:

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, leave=False)
        for x,y in train_bar:

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

            preds = out.argmax(dim=1) if out.ndim > 1 else (out > 0.5).long()
            correct += (preds == y).sum().item()
            total += y.size(0)

            train_bar.set_description(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # ------ Validation ------
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for x,y in val_loader:
                    x, y = x.to(device), y.to(device)

                    pred = model(x)
                    val_loss = criterion(pred, y)
                    val_running += val_loss.item() * x.size(0)

                    preds = pred.argmax(dim=1) if pred.ndim > 1 else (pred > 0.5).long()
                    val_correct += (preds == y).sum().item()
                    val_total += y.size(0)

            val_loss = val_running / len(val_loader.dataset)
            val_acc = val_correct / val_total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}| Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

    return history

def linear_scheduler(optimizer, total_epochs: int):
    def lr_lambda(epoch):
        # decay factor from 1 → 0 linearly
        return 1 - (epoch / max(total_epochs - 1, 1))
    
    return LambdaLR(optimizer, lr_lambda)
