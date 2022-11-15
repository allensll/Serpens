import os
import torch
import torch.nn as nn

from .layers import PConv2d, PLinear

__all__ = ['minionn']


class MiniONN(nn.Module):

    def __init__(self, relu, n=1):
        super(MiniONN, self).__init__()

        self.feature = nn.Sequential(
            PConv2d(3, 64, 3, 1, 0, n=n),
            relu,
            PConv2d(64, 64, 3, 1, 0, n=n),
            relu,
            nn.AvgPool2d(2),
            PConv2d(64, 64, 3, 1, 0, n=n),
            relu,
            PConv2d(64, 64, 3, 1, 0, n=n),
            relu,
            nn.AvgPool2d(2),
            PConv2d(64, 64, 3, 1, 0, n=n),
            relu,
            PConv2d(64, 64, 1, 1, 0, n=n),
            relu,
            PConv2d(64, 16, 1, 1, 0, n=n),
        )
        self.relu7 = relu
        self.linear = PLinear(144, 10, n=n)

    def forward(self, x):
        x = self.feature(x)
        x = self.relu7(x)
        x = x.view(-1, 144)
        x = self.linear(x)
        return x


def minionn(relu=nn.ReLU(), num_classes=10, pretrained=False, n=1):
    model = MiniONN(relu=relu, n=n)
    if pretrained:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pretrained/minionn.pth')
        state_dict = torch.load(data_path, map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict)
    return model

