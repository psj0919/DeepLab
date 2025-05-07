import torch
from torch import nn
from attention_module.ECA_module import ECA



class DNL_DA(nn.Module):
    def __init__(self, inplane):
        super(DNL_DA, self).__init__()
        self.layer1_conv = nn.Conv2d(in_channels=inplane, out_channels=inplane, kernel_size=1)
        self.layer2_conv = nn.Conv2d(in_channels=inplane, out_channels=inplane, kernel_size=1)
        self.layer3_conv = nn.Conv2d(in_channels=inplane, out_channels=inplane, kernel_size=1)
        self.layer4_conv = nn.Conv2d(in_channels=inplane, out_channels=inplane, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(3)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, channel, width, height = x.shape

        # attention
        proj_query = self.layer1_conv(x).view(batch_size, channel, -1)
        proj_key = self.layer2_conv(x).view(batch_size, channel, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        # Prevent Overflow
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        #
        attention = self.softmax(energy_new)  #[1, 3, 3]
        #
        proj_query2 = self.layer3_conv(x)
        proj_query2 = self.avg_pool(proj_query2)
        proj_query2 = proj_query2.mean(dim= -1)

        attention = attention + proj_query2

        proj_value = self.layer4_conv(x).view(batch_size, channel, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out

if __name__=='__main__':
    x = torch.randn(1, 3, 256, 256)
    DNL_DA = DNL_DA(3)
    out = DNL_DA(x)