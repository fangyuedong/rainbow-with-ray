import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicNet(nn.Module):
    def __init__(self, shape, na, norm="batch_norm"):
        """
        shape: observation shape
        na:    action number
        """
        super(BasicNet, self).__init__()
        assert norm in ["batch_norm", "group_norm", None]
        self.shape = shape
        self.na = na
        if norm == "batch_norm":
            self.convs = nn.Sequential(
                nn.Conv2d(shape[0], 32, 8, 4, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 4, 2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )            
        elif norm == "group_norm":
            self.convs = nn.Sequential(
                nn.Conv2d(shape[0], 32, 8, 4, bias=False),
                nn.GroupNorm(16, 32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 4, 2, bias=False),
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, bias=False),
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True)
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(shape[0], 32, 8, 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU(inplace=True)
            )
        self.norm = norm

    
    def forward(self, x):
        return self.convs(x)
    
    def _feature_size(self):
        x = torch.zeros((1,) + self.shape, dtype=torch.float)
        y = self.convs(x)
        return np.prod(y.shape)
    