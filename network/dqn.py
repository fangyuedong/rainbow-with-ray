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
        self.shape = shape
        self.apply(_init_weight)

    def forward(self, x):
        usqz = True if x.ndim == len(self.shape) else False
        x = x.unsqueeze(0) if x.ndim == len(self.shape) else x
        assert x.shape[1:] == self.shape, "x.shape{}".format(tuple(x.shape))
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.output(x)
        x = x.squeeze() if usqz else x
        return x

    def action(self, x):
        x = self.forward(x)
        return x.max(x.ndim-1)[1]
    
    def value(self, x, a=None):
        x = self.forward(x)
        assert a is None or x.ndim == a.ndim+1, "x.ndim{} is not compatible with a.ndim{}".format(x.ndim, a.ndim)
        return x.max(x.ndim-1)[0] if a is None else x.gather(x.ndim-1, a.unsqueeze(a.ndim)).squeeze(a.ndim)

    # def loss_fn(self, x, target, IS=None):
    #     if IS is None:
    #         return F.smooth_l1_loss(x, target), (x - target).abs().detach()
    #     else:
            return torch.mean(F.smooth_l1_loss(x, target, reduction="none") * IS), (x - target).abs().detach()
    @staticmethod
    def td_err(x, target):
        return (x - target).abs()
    