import os
import torch
import torch.nn as nn

from .layers import PConv2d, PLinear


class LeNet5(nn.Module):
    def __init__(self, relu, pool, bias=True, n=1):
        super(LeNet5, self).__init__()
        self.conv1 = PConv2d(1, 6, 5, 1, 0, bias=bias, n=n)
        self.pool1 = pool
        self.relu1 = relu
        self.conv2 = PConv2d(6, 16, 5, 1, 0, bias=bias, n=n)
        self.pool2 = pool
        self.relu2 = relu

        self.full1 = PLinear(4*4*16, 120, bias=bias, n=n)
        self.relu3 = relu
        self.full2 = PLinear(120, 10, bias=bias, n=n)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.pool1(x))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = x.view(-1, 4*4*16)
        x = self.full2(self.relu3(self.full1(x)))
        return x


def lenet5(relu=nn.ReLU(), pool=nn.MaxPool2d(2), num_classes=10, pretrained=False, n=1):
    model = LeNet5(relu=relu, pool=pool, n=n)
    if pretrained:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pretrained/lenet5.pth')
        state_dict = torch.load(data_path)['state_dict']
        model.load_state_dict(state_dict)
    return model
