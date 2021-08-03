# 不带参数的层
import torch
import torch.nn.functional as F
from torch import nn
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X - X.mean()