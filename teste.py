from trainer import set_seed
from Grids.data import get_MNIST_dataloader
from collections import Counter

set_seed(42)

train_loader, test_loader = get_MNIST_dataloader(batch_size=1)

# --- Count occurrences per class ---
train_counts = Counter()
test_counts = Counter()

for _, label in train_loader:
    train_counts[int(label.item())] += 1

for _, label in test_loader:
    test_counts[int(label.item())] += 1

# --- Print results in formatted table form ---
print("MNIST Class Distribution:")
print("-" * 40)
print(f"{'Class':<10}{'Train Count':<15}{'Test Count'}")

total_train = 0
total_test = 0

for cls in range(10):
    tr = train_counts[cls]
    te = test_counts[cls]
    total_train += tr
    total_test += te
    print(f"{cls:<10}{tr:<15}{te}")

print("-" * 40)
print(f"{'TOTAL':<10}{total_train:<15}{total_test}")
