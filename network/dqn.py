import torch
import torch.nn as nn
import torch.nn.functional as F
from network.backbone import BasicNet

def _init_weight(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)

class DQN(nn.Module):
    def __init__(self, shape, na, backbone=BasicNet):
        super(DQN, self).__init__()
        self.backbone = backbone(shape, na)
        self.fc = nn.Sequential(
            nn.Linear(self.backbone._feature_size(), 512),
            nn.ReLU(inplace=True)
        )
        self.output = nn.Linear(512, na)
        self.apply(_init_weight)

    def forward(self, x):
        assert x.shape[1:] == self.shape
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.output(x)
        return x

    def action(self, x):
        return self.forward(x).max(1, True)[1]
    
    def value(self, x, a=None):
        assert a is None or a.shape == (x.shape[0], 1)
        return self.forward(x).max(1, True)[0] if a is None else self.forward(x).gather(1, a)

    def loss_fn(self, x, target):
        return F.smooth_l1_loss(x, target)
    