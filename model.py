import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Net(nn.Module):
    def __init__(self, num_channels=5, num_classes=3):
        super().__init__()
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv7 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Downsample
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(F.max_pool2d(x1, 2)))
        x3 = F.relu(self.conv3(F.max_pool2d(x2, 2)))
        x4 = F.relu(self.conv4(F.max_pool2d(x3, 2)))
        # Upsample and concatenate
        x5 = F.relu(self.conv5(torch.cat([self.upconv1(x4), x3], 1)))
        x6 = F.relu(self.conv6(torch.cat([self.upconv2(x5), x2], 1)))
        x7 = F.relu(self.conv7(torch.cat([self.upconv3(x6), x1], 1)))
        out = self.final_conv(x7)
        return out