import torch
import torch.nn as nn
from math import log
from attention_module.ECA_module import ECA


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding,
                                     dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone =='mobilenet':
            inplanes = 320
        else:
            inplanes = 2048

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise  NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # ECA
        self.eca_x = nn.Conv2d(2048, 256, 1, bias=False)
        self.t = int (abs((log(2048, 2) + 1) / 2))
        self.k = self.t if self.t % 2 else self.t + 1
        self.eca_module = ECA(self.k)
        # DA_ECA
        # self.t = int (abs((log(2048, 2) + 1) / 2))
        # self.k = self.t if self.t % 2 else self.t + 1
        # self.eca_module = DA_ECA(self.k)

        self._init_weight()

    def forward(self, x):
        # x.shape = [8, 2048, 16, 16]
        x1 = self.aspp1(x)  # x1.shape = [8, 256, 16, 16]
        x2 = self.aspp2(x)  # x2.shape = [8, 256, 16, 16]
        x3 = self.aspp3(x)  # x3.shape = [8, 256, 16, 16]
        x4 = self.aspp4(x) # x4.shape = [8, 256, 16, 16]
        # ECA
        eca_x = self.eca_x(x)
        x5 = self.eca_module(eca_x) # x5.shape = [8, 256, 16, 16]
        # DA_ECA
        # x5 = self.da_eca_module(x) # x5.shape = [8, 256, 16, 16]

        # x5 = self..global_avg_pool(x)
        # x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        return self.dropout(x)


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)



if __name__=='__main__':
    x = torch.randn(8, 2048, 16, 16)
    model = build_aspp('resnet', 16, nn.BatchNorm2d)
    out = model(x)