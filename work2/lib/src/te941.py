import torch as _torch
from torch import nn as _nn


class NN1(_nn.Module):
    def __init__(self):
        super().__init__()

        self.body = _nn.Sequential(
            _nn.Conv2d(1, 32, kernel_size=3, stride=1, padding='same'),
            _nn.ReLU(),
            _nn.Conv2d(32, 32, kernel_size=2, stride=2),
            _nn.ReLU(),
        )

        self.head = _nn.Sequential(
            _nn.Linear(32 * 14 * 14, 100),
            _nn.ReLU(),
            _nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.body(x)
        x = _torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.head(x)
        return x

class NN2(_nn.Module):
    def __init__(self):
        super().__init__()

        self.body = _nn.Sequential(
            _nn.Conv2d(1, 32, kernel_size=5, stride=1, padding='same'),
            _nn.ReLU(),
            _nn.Conv2d(32, 32, kernel_size=2, stride=2),
            _nn.ReLU(),
        )

        self.head = _nn.Sequential(
            _nn.Linear(32 * 14 * 14, 100),
            _nn.ReLU(),
            _nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.body(x)
        x = _torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.head(x)
        return x

class NN3(_nn.Module):
    def __init__(self):
        super().__init__()

        self.body = _nn.Sequential(
            _nn.Conv2d(1, 64, kernel_size=5, stride=1, padding='same'),
            _nn.ReLU(),
            _nn.Conv2d(64, 64, kernel_size=2, stride=2),
            _nn.ReLU(),
        )

        self.head = _nn.Sequential(
            _nn.Linear(64 * 14 * 14, 100),
            _nn.ReLU(),
            _nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.body(x)
        x = _torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.head(x)
        return x

class NN4(_nn.Module):
    def __init__(self):
        super().__init__()

        self.body = _nn.Sequential(
            _nn.Conv2d(1, 32, kernel_size=3, stride=1, padding='same'),
            _nn.ReLU(),
            _nn.Conv2d(32, 32, kernel_size=2, stride=2),
            _nn.ReLU(),
            _nn.Conv2d(32, 64, kernel_size=5, stride=1, padding='same'),
            _nn.ReLU(),
            _nn.Conv2d(64, 64, kernel_size=2, stride=2),
            _nn.ReLU(),
        )

        self.head = _nn.Sequential(
            _nn.Linear(64 * 7 * 7, 100),
            _nn.ReLU(),
            _nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.body(x)
        x = _torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.head(x)
        return x

class NN5(_nn.Module):
    def __init__(self):
        super().__init__()

        self.body = _nn.Sequential(
            _nn.Conv2d(1, 32, kernel_size=3, stride=1, padding='same'),
            _nn.ReLU(),
            _nn.Conv2d(32, 32, kernel_size=2, stride=2),
            _nn.ReLU(),
            _nn.Conv2d(32, 64, kernel_size=5, stride=1, padding='same'),
            _nn.ReLU(),
            _nn.Conv2d(64, 64, kernel_size=2, stride=2),
            _nn.ReLU(),
        )

        self.head = _nn.Sequential(
            _nn.Conv2d(64, 128, kernel_size=1, stride=1),
            _nn.ReLU(),
            _nn.Conv2d(128, 10, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        x = _torch.mean(x, dim=[-2, -1])
        return x
