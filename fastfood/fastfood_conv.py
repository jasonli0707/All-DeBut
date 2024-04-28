import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt

from sklearn_extra.utils._cyfht import fht2 as cyfht
from sklearn.utils import check_array
from scipy.linalg import hadamard
import time

class LinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input
        output = check_array(output.cpu().numpy(), order='C')
        cyfht(output)
        output = torch.from_numpy(output).cuda()
        return output

    @staticmethod
    def backward(ctx, grad_output): 
        input = ctx.saved_variables
        grad_input = None  #grad_bias
        grad_input = grad_output
        grad_input = check_array(grad_input.cpu().numpy(), order='C')
        cyfht(grad_input)
        grad_input = torch.from_numpy(grad_input).cuda()
        return grad_input

class Fastfood_2dConv(nn.Module):
    """Generate fastfood to CONV layers"""

    def __init__(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0, dilation = 1,
                bias=True):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.in_size = self.input_channel * self.kernel_size[0] * self.kernel_size[1]
        self.out_size = self.output_channel

        m = int(math.ceil(math.log2(self.in_size)))
        self.m = m
        size = self.in_size_extended = 1 << m  # Will zero-pad input if in_size is not a power of 2
        self.nstack = int(math.ceil(self.out_size / self.in_size_extended))
        n = self.nstack * size  # new output size
        
        # new parameters
        self.P = torch.randperm(n)
        self.B = nn.Parameter(torch.from_numpy(np.random.normal(0, 0.1, size = (self.nstack, self.in_size_extended))).float())
        self.G = nn.Parameter(torch.from_numpy(np.random.normal(0, 0.1, size = (self.nstack, self.in_size_extended))).float())
        self.S = nn.Parameter(torch.from_numpy(np.random.normal(0, 0.1, size = (self.nstack, self.in_size_extended))).float())
        
        if bias:
            bias_shape = (self.out_size, ) 
            self.bias = nn.Parameter(torch.Tensor(*bias_shape))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def _diag_first(self, B, input_x):
        batch_size = input_x.shape[0]
        output = B * input_x.view(1, batch_size, 1, self.in_size_extended)
        output = output.view(batch_size * self.nstack, self.in_size_extended)
        return output
    
    def _diag_secend(self, G, input_x):
        batch_size = input_x.shape[0]
        output = G.view(1, -1) * input_x.view(batch_size, self.in_size_extended * self.nstack)
        return output.view(batch_size * self.nstack, self.in_size_extended)
    
    def _diag_third(self, S, input_x):
        batch_size = input_x.shape[0]
        output = S.view(1, -1) * input_x.view(batch_size, self.in_size_extended * self.nstack)
        return output

    def _perm_trans(self, P, input_x):
        batch_size = input_x.shape[0]
        output = input_x[:, P]
        return output
    
    def _hadamard_trans(self, input_x):
        H = torch.from_numpy(hadamard(self.in_size_extended)).float().cuda()
        output = torch.mm(H, input_x.transpose(1, 0))
        return output.transpose(1, 0)

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Parameters:
            input: (batch, *, in_size) if real or (batch, *, in_size, 2) if complex
        Return:
            output: (batch, *, out_size) if real or (batch, *, out_size, 2) if complex
        """
        output = self.pre_process(input)
        batch_size = output.shape[0]
        output = self._diag_first(self.B, output)
        output = LinearFunction.apply(output)

        output = output.reshape(batch_size, -1)
        output = self._perm_trans(self.P, output)
        output = self._diag_secend(self.G, output)

        output = LinearFunction.apply(output)
        output = output.reshape(batch_size, -1)
        output = self._diag_third(self.S, output) 

        return self.post_process(input, output)

    def pre_process(self, input):
        batch, c, h, w = input.shape
        h_out = (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (h + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        
        input_patches = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride).view(
            batch, c, self.kernel_size[0] * self.kernel_size[1], h_out * w_out)
        output = input_patches.permute(0, 3, 2, 1).reshape(batch * h_out * w_out, self.kernel_size[0] * self.kernel_size[1] * c)
        output = F.pad(output, (0, self.in_size_extended - self.in_size, 0, 0), "constant", value = 0)
        return output
   
    def post_process(self, input, output):
        batch, c, h, w = input.shape
        h_out = (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (h + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        output = output[:, :self.out_size]
        if self.bias is not None:
            output += self.bias
        output = output.float()
  
        return output.view(batch, h_out * w_out, self.output_channel).transpose(1, 2).view(batch, self.output_channel, h_out, w_out)

    def extra_repr(self):
        s = 'input_channel={}, output_channel={}, kernel_size = {}, stride = {}, padding = {}, delation = {}, bias={}'.format(self.input_channel, 
                self.output_channel, self.kernel_size, self.stride, self.padding, self.dilation, self.bias is not None)
        return s


class Fastfood_1dConv(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0, dilation = 1,
                bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.in_size = input_channel * kernel_size
        self.out_size = output_channel

        m = int(math.ceil(math.log2(self.in_size)))
        self.m = m
        size = self.in_size_extended = 1 << m  # Will zero-pad input if in_size is not a power of 2
        self.nstack = int(math.ceil(self.out_size / self.in_size_extended))
        n = self.nstack * size  # new output size
        
        # new parameters
        self.P = torch.randperm(n)
        self.B = nn.Parameter(torch.from_numpy(np.random.normal(0, 0.1, size = (self.nstack, self.in_size_extended))).float())
        self.G = nn.Parameter(torch.from_numpy(np.random.normal(0, 0.1, size = (self.nstack, self.in_size_extended))).float())
        self.S = nn.Parameter(torch.from_numpy(np.random.normal(0, 0.1, size = (self.nstack, self.in_size_extended))).float())
        
        if bias:
            bias_shape = (self.out_size, ) 
            self.bias = nn.Parameter(torch.Tensor(*bias_shape))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def _diag_first(self, B, input_x):
        batch_size = input_x.shape[0]
        output = B * input_x.view(1, batch_size, 1, self.in_size_extended)
        output = output.view(batch_size * self.nstack, self.in_size_extended)
        return output
    
    def _diag_secend(self, G, input_x):
        batch_size = input_x.shape[0]
        output = G.view(1, -1) * input_x.view(batch_size, self.in_size_extended * self.nstack)
        return output.view(batch_size * self.nstack, self.in_size_extended)
    
    def _diag_third(self, S, input_x):
        batch_size = input_x.shape[0]
        output = S.view(1, -1) * input_x.view(batch_size, self.in_size_extended * self.nstack)
        return output

    def _perm_trans(self, P, input_x):
        batch_size = input_x.shape[0]
        output = input_x[:, P]
        return output
    
    def _hadamard_trans(self, input_x):
        H = torch.from_numpy(hadamard(self.in_size_extended)).float().cuda()
        output = torch.mm(H, input_x.transpose(1, 0))
        return output.transpose(1, 0)

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Parameters:
            input: (batch, *, in_size) if real or (batch, *, in_size, 2) if complex
        Return:
            output: (batch, *, out_size) if real or (batch, *, out_size, 2) if complex
        """
        output = self.pre_process(input)
        batch_size = output.shape[0]
        output = self._diag_first(self.B, output)
        output = LinearFunction.apply(output)

        output = output.reshape(batch_size, -1)
        output = self._perm_trans(self.P, output)
        output = self._diag_secend(self.G, output)

        output = LinearFunction.apply(output)
        output = output.reshape(batch_size, -1)
        output = self._diag_third(self.S, output) 

        return self.post_process(input, output)

    def pre_process(self, input):
        batch, c_in, l = input.shape
        input = input.view(batch, c_in, 1, l)
        l_out = (l + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        inp_unf = F.unfold(input, kernel_size=(1, self.kernel_size), padding=(0, self.padding)) # (batch, kernel_size*c_in, l_out)
        inp_unf = inp_unf.view(batch, c_in, self.kernel_size, l_out) # shape = (b, c_in, k, l_o)
        inp_unf = inp_unf.permute(0, 3, 2, 1) # shape = (b, l_o, k, c_in)
        output = inp_unf.reshape(batch * l_out, self.kernel_size * c_in) # shape = (b*l_o, k*c_in)
        output = F.pad(output, (0, self.in_size_extended - self.in_size, 0, 0), "constant", value = 0)
        return output
   
    def post_process(self, input, output):
        batch, c, l = input.shape
        l_out = (l + 2 * self.padding - self.dilation * (self.kernel_size- 1) - 1) // self.stride + 1
        output = output[:, :self.out_size]
        if self.bias is not None:
            output += self.bias
        output = output.float()
  
        return output.view(batch, l_out, self.output_channel).transpose(1, 2)

    def extra_repr(self):
        s = 'input_channel={}, output_channel={}, kernel_size = {}, stride = {}, padding = {}, delation = {}, bias={}'.format(self.input_channel, 
                self.output_channel, self.kernel_size, self.stride, self.padding, self.dilation, self.bias is not None)
        return s

if __name__ == "__main__":
    x = torch.randn(10, 3, 100).cuda()
    conv = nn.Conv1d(3, 64, 3, padding=1).cuda()
    print(sum([p.numel()for p in conv.parameters()]))
    svd_conv = Fastfood_1dConv(3, 64, 3, padding=1).cuda()
    for n, p in svd_conv.named_parameters():
        print(n, p.numel())
    print(sum([p.numel()for p in svd_conv.parameters()]))
    actual = conv(x)
    result = svd_conv(x)
    print(actual.shape)
    print(result.shape)