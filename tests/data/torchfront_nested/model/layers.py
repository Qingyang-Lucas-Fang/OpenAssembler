import torch
from torch import nn
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 8)
    def forward(self, x):
        return torch.sin(self.projection(x))
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.first, self.second = Block(), Block()
    def forward(self, x):
        return self.second(self.first(x))
