import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
import correlation_cuda
import torch.nn.functional as F
import torch.nn as nn


class CorrelationFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        ctx.pad_size = 4
        ctx.kernel_size = 1
        ctx.max_displacement = 4
        ctx.stride1 = 1
        ctx.stride2 = 1
        ctx.corr_multiply = 1

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()
            correlation_cuda.forward(input1, input2, rbot1, rbot2, output,
                                     ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                                      ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)

        return grad_input1, grad_input2


class Correlation(Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):

        result = CorrelationFunction.apply(input1, input2)
        return result



def compute_cost_volume(feat1, feat2, param_dict):
    """
    only implemented for:
        kernel_size = 1
        stride1 = 1
        stride2 = 1
    """

    max_disp = param_dict["max_disp"]

    _, _, height, width = feat1.size()
    num_shifts = 2 * max_disp + 1
    feat2_padded = F.pad(feat2, (max_disp, max_disp, max_disp, max_disp), "constant", 0)

    cost_list = []
    for i in range(num_shifts):
        for j in range(num_shifts):
            corr = torch.mean(feat1 * feat2_padded[:, :, i:(height + i), j:(width + j)], axis=1, keepdims=True)
            cost_list.append(corr)
    cost_volume = torch.cat(cost_list, axis=1)
    return cost_volume


def my_compute_cost(x1, x2, H=None, W=None, channel=None, d=9):
    if H is None or W is None or channel is None:
        _, channel, H, W = x1.shape
    # choice 1: use torch.nn.functional.unfold
    x2_patches = F.unfold(x2, kernel_size=d, stride=1, padding=d // 2)
    x2_patches = x2_patches.view(-1, channel, d, d, H, W)

    x1_reshape = x1.view(-1, channel, 1, 1, H, W)
    x1_dot_x2 = torch.mul(x1_reshape, x2_patches)
    cost_volume = torch.sum(x1_dot_x2, dim=1)
    cost_volume = cost_volume.view(-1, d * d, H, W)
    return cost_volume

