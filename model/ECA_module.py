import torch
from torch import nn
from math import log

class ECA(nn.Module):
    def __init__(self):
        super(ECA, self).__init__()
        self.gamma = 2
        self.b = 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.size()

        t = int (abs((log(C, 2) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)

        out = self.avg_pool(x)
        out = conv(out.squeeze(-1).transpose(-1, -2))
        out = out.transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)

        return x * out.expand_as(x)


if __name__== '__main__':
    x = torch.randn(1, 3, 256, 256)
    ECA = ECA()
    out = ECA(x)