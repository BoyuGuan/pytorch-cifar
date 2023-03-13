import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class DNN_CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_block_1 = nn.Sequential(*[
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        ])
        self.flatten = lambda inp: torch.flatten(inp, 1)
        self.Linear1 =  nn.Linear(32 * 16 * 16, 16)

        self.Linear2 = nn.Sequential(*[
            nn.Linear(16, 24576),
            nn.Linear(24576, 16)
        ])
        self.Linear3 = nn.Sequential(*[
            nn.Linear(16, 24576),
            nn.Linear(24576, 16)
        ])
        self.out = nn.Linear(16, 10)


    def forward(self, X):
        X = self.cnn_block_1(X)
        X = self.flatten(X)
        X = self.Linear1(X)
        X = self.Linear2(X)
        X = self.Linear3(X)
        X = self.out(X)
        return X


class DNN_CIFAR10Model_checkpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_block_1 = nn.Sequential(*[
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        ])
        self.flatten = lambda inp: torch.flatten(inp, 1)
        self.Linear1 =  nn.Linear(32 * 16 * 16, 16)

        self.Linear2 = nn.Sequential(*[
            nn.Linear(16, 24576),
            nn.Linear(24576, 16)
        ])
        self.Linear3 = nn.Sequential(*[
            nn.Linear(16, 24576),
            nn.Linear(24576, 16)
        ])
        self.out = nn.Linear(16, 10)


    def forward(self, X):
        X = self.cnn_block_1(X)
        X = self.flatten(X)
        X = self.Linear1(X)
        X = checkpoint(self.Linear2, X)
        X = checkpoint(self.Linear3, X)
        X = self.out(X)
        return X


class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_block_1 = nn.Sequential(*[
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        ])
        self.cnn_block_2 = nn.Sequential(*[
            nn.Conv2d(64, 4096, 3, padding=1),
            nn.Conv2d(4096, 64, 3, padding=1),
        ])
        self.cnn_block_3 = nn.Sequential(*[
            nn.Conv2d(64, 4096, 3, padding=1),
            nn.Conv2d(4096, 32, 3, padding=1),
        ])

        self.flatten = lambda inp: torch.flatten(inp, 1)
        self.head = nn.Sequential(*[
            nn.Linear(32 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        ])

    def forward(self, X):
        X = self.cnn_block_1(X)
        X = self.cnn_block_2(X)
        X = self.cnn_block_3(X)
        X = self.flatten(X)
        X = self.head(X)
        return X
    

class CIFAR10Model_checkpoint(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn_block_1 = nn.Sequential(*[
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        ])
        self.cnn_block_2 = nn.Sequential(*[
            nn.Conv2d(64, 4096, 3, padding=1),
            nn.Conv2d(4096, 64, 3, padding=1),
        ])
        self.cnn_block_3 = nn.Sequential(*[
            nn.Conv2d(64, 4096, 3, padding=1),
            nn.Conv2d(4096, 32, 3, padding=1),
        ])

        self.flatten = lambda inp: torch.flatten(inp, 1)
        self.head = nn.Sequential(*[
            nn.Linear(32 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        ])

    def forward(self, X):
        X = self.cnn_block_1(X)
        X = checkpoint(self.cnn_block_2, X)
        X = checkpoint(self.cnn_block_3, X)
        X = self.flatten(X)
        X = self.head(X)
        return X
    
    