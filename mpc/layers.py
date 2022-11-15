import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import timer


class ReLU2PC(nn.Module):
    def __init__(self, agt=None):
        super(ReLU2PC, self).__init__()
        self.agt = agt

    def forward(self, x):
        shape = x.shape
        with timer('relu2pc'):
            if self.agt and not self.training:
                x = x.flatten().tolist()
                x = self.agt.relu(x, False)
                x = torch.Tensor(x).reshape(shape)
        return x


class ReLUMPC(nn.Module):
    def __init__(self, agt=None):
        super(ReLUMPC, self).__init__()
        self.agt = agt
        self.mask = None

    def forward(self, x):
        # print(x.shape)
        if self.agt and not self.training:
            with timer('mask + shuffle'):
                # 1 mask
                k = torch.rand_like(x) + 0.5
                a = x * k
                # 2 shuffle
                if len(x.shape) == 2:
                    idx = torch.randperm(a.nelement())
                    a = a.view(-1)[idx]
                else: # method 2
                    sn = x.shape[0] * x.shape[1]
                    idx = torch.randperm(sn)
                    a = a.view(sn, -1)[idx]

            # 3 central server
            sl = self.agt.send_output(('relu', a.half()))
            print('send {:.4f}MB'.format(sl/1014.0/1024.0))
            # a = self.agt.recv_input().float()
            a = self.agt.recv_input()
            with timer('unshuffle + remask'):
                # 4 unshuffle
                re_idx = torch.empty_like(idx)
                re_idx[idx] = torch.arange(idx.nelement(), device=idx.device)
                a = a[re_idx].view(x.size())
                # 5 de-mask
                x = a / k
        else:
            x = F.relu(x)
        return x


class MaxPool2d2PC(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, agt=None):
        super(MaxPool2d2PC, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.agt = agt
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=self.stride, padding=padding, dilation=dilation)

    def forward(self, x):
        shape = x.shape
        with timer('maxpool2d2pc'):
            if self.agt and not self.training:
                x = self.unfold(x.view(shape[0]*shape[1], 1, shape[2], shape[3]))
                x = x.permute(1, 0, 2).flatten().tolist()
                with timer('--- net'):
                    x = self.agt.maxpool2d(x, self.kernel_size**2)
                x = torch.Tensor(x).reshape(shape[0], shape[1], int(shape[2]/self.stride), int(shape[3]/self.stride))
        return x


class MaxPool2dMPC(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, agt=None):
        super(MaxPool2dMPC, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.agt = agt

    def forward(self, x):
        # print(x.shape)
        if self.agt and not self.training:
            with timer('mask + shuffle'):
                # 1 extract
                shape = F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation).shape
                s = self.kernel_size[0]*self.kernel_size[1] if type(self.kernel_size) is tuple else self.kernel_size**2
                x = x.view(-1, x.size(2), x.size(3)).unsqueeze(1)
                a = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
                a = a.permute(0, 2, 1).contiguous().view(-1, s)
                # 2 mask
                # torch.manual_seed(self.agt.rand_seed)
                k = torch.rand(a.size(0), 1) + 0.5
                a = a * k.expand(a.size())
                # 3 shuffle
                idx = torch.randperm(a.size(0))
                a = a[idx]
            # 4 server
            sl = self.agt.send_output(('maxpool', a.half()))
            print('send {:.4f}MB'.format(sl/1014.0/1024.0))
            a = self.agt.recv_input().float()
            with timer('unshuffle + remask'):
                # 5 unshuffle
                re_idx = torch.empty_like(idx)
                re_idx[idx] = torch.arange(idx.nelement(), device=idx.device)
                a = a[re_idx]
                # 6 de-mask
                a = a / k.view(-1)
                x = a.reshape(shape)
        else:
            x = F.max_pool2d(x, self.kernel_size, self.stride,
                             self.padding, self.dilation)
        return x
