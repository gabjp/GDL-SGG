import torch.nn as nn
import torch.nn.functional as F
from kornia.filters.blur_pool import BlurPool2D


class CNN(nn.Module):
    """
    Input (MNIST): [B, 1, 28, 28]
    """

    def __init__(self, conv1=32, conv2=64, conv3=128, conv4=128, num_classes=10):
        super().__init__()

        # --- BLOCK 1 ---
        # Conv keeps spatial size: SAME padding (circular) with kernel 3x3
        # Output: [B, 32, 28, 28]
        self.conv1 = nn.Conv2d(1, conv1, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        self.bn1 = nn.BatchNorm2d(conv1)

        # Conv again, still spatially same size
        # Output: [B, 64, 28, 28]
        self.conv2 = nn.Conv2d(conv1, conv2, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        self.bn2 = nn.BatchNorm2d(conv2)

        # BlurPool (anti-aliased downsample)
        # Downsampling by stride=2 halves H and W
        # Output: [B, conv2, 14, 14]
        self.down1 = BlurPool2D(conv2, stride=2)

        # --- BLOCK 2 ---
        # Conv preserves size again
        # Output: [B, 128, 14, 14]
        self.conv3 = nn.Conv2d(conv2, conv3, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        self.bn3 = nn.BatchNorm2d(conv3)

        # Another SAME conv
        # Output: [B, conv3, 14, 14]
        self.conv4 = nn.Conv2d(conv3, conv4, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        self.bn4 = nn.BatchNorm2d(conv4)

        # Second anti-aliased downsample
        # Output: [B, conv4, 7, 7]
        #self.down2 = BlurPool2D(conv4, stride=2)

        # Global average pooling:
        # Reduces 7x7 feature maps → 1x1 (by averaging)
        # Output: [B, conv4]
        self.classifier = nn.Linear(conv4, num_classes)


    def forward(self, x):
        # x: [B, 1, 28, 28]

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        #x = F.relu(self.bn1(self.conv1(x)))  
        # → [B, 32, 28, 28]

        x = F.relu(self.bn2(self.conv2(x)))
        # → [B, 64, 28, 28]

        x = self.down1(x)
        # → [B, 64, 14, 14]

        x = F.relu(self.bn3(self.conv3(x)))
        # → [B, 128, 14, 14]

        x = F.relu(self.bn4(self.conv4(x)))
        # → [B, 128, 14, 14]

        #x = self.down2(x)
        # → [B, 128, 7, 7]

        # Global spatial averaging → invariant representation
        x = x.mean(dim=(-1, -2))
        # → [B, 128]

        return self.classifier(x)
        # → [B, 10]
