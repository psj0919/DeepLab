import torch.nn as nn
import torch


def conv2d(in_channel, out_channel, kernel_size):
    layers = [
        nn.Conv2d(
            in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False
        ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)


def conv1d(in_channel, out_channel):
    layers = [
        nn.Conv1d(in_channel, out_channel, 1, bias=False),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)

class OCR(nn.Module):
    def __init__(self, n_class, feat_channel=[1024, 2048]):
        super(OCR, self).__init__()
        ch16, ch32 = feat_channel
        self.L = nn.Conv2d(ch16, n_class, 1)
        self.X = conv2d(ch32, 512, 3)

        self.phi = conv1d(512, 256)
        self.psi = conv1d(512, 256)
        self.delta = conv1d(512, 256)
        self.rho = conv1d(256, 512)
        self.g = conv2d(512 + 512, 512, 1)

        self.out = nn.Conv2d(512, n_class, 1)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x, low_level_feature):
        X = self.X(x)
        L = self.L(low_level_feature)

        batch, n_class, height, width = L.shape
        l_flat = L.view(batch, n_class, -1)
        # M: NKL
        M = torch.softmax(l_flat, -1)
        channel = X.shape[1]
        X_flat = X.view(batch, channel, -1)
        # f_k: NCK
        f_k = (M @ X_flat.transpose(1, 2)).transpose(1, 2)

        # query: NKD
        query = self.phi(f_k).transpose(1, 2)
        # key: NDL
        key = self.psi(X_flat)
        logit = query @ key
        # attn: NKL
        attn = torch.softmax(logit, 1)

        # delta: NDK
        delta = self.delta(f_k)
        # attn_sum: NDL
        attn_sum = delta @ attn
        # x_obj = NCHW
        X_obj = self.rho(attn_sum).view(batch, -1, height, width)

        concat = torch.cat([X, X_obj], 1)
        X_bar = self.g(concat)
        out = self.out(X_bar)

        return out

if __name__=='__main__':
    input = torch.rand(1, 1024, 16, 16)   # bottleneck3
    input2 = torch.rand(1, 2048, 16, 16)  # bottleneck4
    OCR = OCR(21, [1024, 2048])
    out = OCR(input2, input)

