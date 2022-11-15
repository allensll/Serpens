import torch
import torch.nn as nn
import torch.nn.functional as F


class PBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, n=1):
        super(PBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

        self.n = n
        self.offset = None

    def forward(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        else:
            return F.batch_norm(
                input, self.running_mean/self.n, self.running_var, self.weight, self.bias/self.n,
                False, exponential_average_factor, self.eps)

        # if self.training:
        #     bn_training = True
        # else:
        #     bn_training = (self.running_mean is None) and (self.running_var is None)
        # return F.batch_norm(
        #     input,
        #     # If buffers are not to be tracked, ensure that they won't be updated
        #     self.running_mean / self.n
        #     if not self.training or self.track_running_stats
        #     else None,
        #     self.running_var if not self.training or self.track_running_stats else None,
        #     self.weight,
        #     self.bias/self.n,
        #     bn_training,
        #     exponential_average_factor,
        #     self.eps,
        # )


class PConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, n=1):
        super(PConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                      padding, dilation, groups, bias)

        self.n = n

    def forward(self, input):
        if self.training or self.bias is None:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(input, self.weight, self.bias/self.n, self.stride, self.padding, self.dilation, self.groups)


class PLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, n=1):
        super(PLinear, self).__init__(in_features, out_features, bias)

        self.n = n

    def forward(self, input):
        if self.training or self.bias is None:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(input, self.weight, self.bias/self.n)


if __name__ == '__main__':

    pass
    # # test BatchNorm2d
    # input = torch.rand(2, 3, 2, 2)
    # mask = torch.rand(2, 3, 2, 2)
    # input_trans = input - mask
    #
    # l1 = nn.BatchNorm2d(3)
    # l2 = PBatchNorm2d(3)
    # l2.running_var, l2.running_mean = l1.running_var, l1.running_mean = torch.rand(3), torch.rand(3)
    # l2.weight, l2.bias = l1.weight, l1.bias
    # l1.eval()
    # l2.eval()
    #
    # output = l1(input)
    # output_trans = l2(input_trans)
    # output_mask = l2(mask)
    # print(output)
    # print(output_trans + output_mask)

    # # test Conv2d
    # input = torch.rand(2, 3, 5, 5)
    # mask = torch.rand(2, 3, 5, 5)
    # input_trans = input - mask
    #
    # l1 = nn.Conv2d(3, 2, 4)
    # l2 = PConv2d(3, 2, 4)
    # l2.weight, l2.bias = l1.weight, l1.bias
    # l1.eval()
    # l2.eval()
    #
    # output = l1(input)
    # output_trans = l2(input_trans)
    # output_mask = l2(mask)
    # print(output)
    # print(output_trans + output_mask)
    #
    # # test Liner
    # input = torch.rand(2, 10)
    # mask = torch.rand(2, 10)
    # input_trans = input - mask
    #
    # l1 = nn.Linear(10, 2)
    # l2 = PLinear(10, 2)
    # l2.weight, l2.bias = l1.weight, l1.bias
    # l1.eval()
    # l2.eval()
    #
    # output = l1(input)
    # output_trans = l2(input_trans)
    # output_mask = l2(mask)
    # print(output)
    # print(output_trans + output_mask)
