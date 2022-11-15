# https://github.com/akamaster/pytorch_resnet_cifar10

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .layers import PConv2d, PLinear, PBatchNorm2d


__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, PLinear) or isinstance(m, PConv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, relu,  stride=1, option='A', n=1):
        super(BasicBlock, self).__init__()
        self.conv1 = PConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, n=n)
        self.bn1 = PBatchNorm2d(planes, n=n)
        self.conv2 = PConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, n=n)
        self.bn2 = PBatchNorm2d(planes, n=n)
        self.relu = relu

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    PConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, n=n),
                    PBatchNorm2d(self.expansion * planes, n=n)
                )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, relu, num_classes=10, n=1):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = PConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, n=n)
        self.bn1 = PBatchNorm2d(16, n=n)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], relu, stride=1, n=n)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], relu, stride=2, n=n)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], relu, stride=2, n=n)
        self.linear = PLinear(64, num_classes, n=n)
        self.relu = relu

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, relu, stride, n):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, relu, stride, n=n))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def _resnet(arch, block, layers, pretrained, relu, n=1, **kwargs):
    model = ResNet(block, layers, relu, n=n, **kwargs)
    if pretrained:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pretrained/{}.pth'.format(arch))
        state_dict = torch.load(data_path, map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict)
    return model


def resnet20(relu=nn.ReLU(), pretrained=False, **kwargs):
    return _resnet('resnet20', BasicBlock, [3, 3, 3], pretrained, relu, **kwargs)


def resnet32(relu=nn.ReLU(), pretrained=False, **kwargs):
    return _resnet('resnet32', BasicBlock, [5, 5, 5], pretrained, relu, **kwargs)


def resnet44(relu=nn.ReLU(), pretrained=False, **kwargs):
    return _resnet('resnet44', BasicBlock, [7, 7, 7], pretrained, relu, **kwargs)


def resnet56(relu=nn.ReLU(), pretrained=False, **kwargs):
    return _resnet('resnet56', BasicBlock, [9, 9, 9], pretrained, relu, **kwargs)


def resnet110(relu=nn.ReLU(), pretrained=False, **kwargs):
    return _resnet('resnet110', BasicBlock, [18, 18, 18], pretrained, relu, **kwargs)


def resnet1202(relu=nn.ReLU(), pretrained=False, **kwargs):
    return _resnet('resnet1202', BasicBlock, [200, 200, 200], pretrained, relu, **kwargs)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))