import torch
import numpy as np
import math
from math import sqrt
from torch import nn
import torch.nn.functional as F
# from torch.autograd.function import once_differentiable

from sklearn_extra.utils._cyfht import fht2 as cyfht
from sklearn.utils import check_array
from scipy.linalg import hadamard
from torch.autograd import gradcheck

# from deep_fried_convnets.fastfood_layers import Hadamard_layer
   
# fastfood_cfg = ['diag', 'wh', 'perm', 'diag', 'wh', 'diag']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input
        output = check_array(output.cpu().numpy(), order='C')
        cyfht(output)
        output = torch.from_numpy(output).cuda()
#         output = input.mm(weight.t())
        # if bias is not None:
        #     output += bias.unsqueeze(0).expand_as(output)
        # print('output after forward:{}'.format(output))
        return output

    @staticmethod
    def backward(ctx, grad_output): 
        input = ctx.saved_variables
        grad_input = None  #grad_bias
        # print('grad_output is:\n {}'.format(grad_output))
        grad_input = grad_output
        grad_input = check_array(grad_input.cpu().numpy(), order='C')
        cyfht(grad_input)
        grad_input = torch.from_numpy(grad_input).cuda()
        # print('grad_input after backward is:\n {}'.format(grad_input))
        
        # grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input

class FastfoodTransform(nn.Module):
    def __init__(self, in_size, out_size, bias=True):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size 
        m = int(math.ceil(math.log2(in_size)))
        self.m = m
        size = self.in_size_extended = 1 << m  # Will zero-pad input if in_size is not a power of 2
        self.nstack = int(math.ceil(out_size / self.in_size_extended))
        n = self.nstack * size  # new output size

        # self.B = nn.Parameter(torch.randn(self.nstack, self.in_size_extended))
        self.P = torch.randperm(n)
        # self.B = nn.Parameter(torch.randn(self.nstack, self.in_size_extended))
        # self.G = nn.Parameter(torch.randn(self.nstack, self.in_size_extended))
        # self.S = nn.Parameter(torch.randn(self.nstack, self.in_size_extended))
        self.B = nn.Parameter(torch.from_numpy(np.random.normal(0, 0.1, size = (self.nstack, self.in_size_extended))).float())
        self.G = nn.Parameter(torch.from_numpy(np.random.normal(0, 0.1, size = (self.nstack, self.in_size_extended))).float())
        self.S = nn.Parameter(torch.from_numpy(np.random.normal(0, 0.1, size = (self.nstack, self.in_size_extended))).float())
        # self.H2 = torch.from_numpy(hadamard(self.in_size_extended)).float().cuda()
        # self.hadamard_layer = Hadamard_layer(self.in_size_extended)

        if bias:
            bias_shape = (out_size, ) 
            self.bias = nn.Parameter(torch.Tensor(*bias_shape))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def _diag_first(self, B, input_x):
        batch_size = input_x.shape[0]
        output = B * input_x.view(1, batch_size, 1, self.in_size_extended)
        output = output.view(batch_size * self.nstack, self.in_size_extended)
        # print('output after the first diag is: {}'.format(output))
        return output
    
    def _diag_secend(self, G, input_x):
        batch_size = input_x.shape[0]
        output = G.view(1, -1) * input_x.view(batch_size, self.in_size_extended * self.nstack)
        # print(output.shape)
        # print('output after the second diag is: {}'.format(output))
        return output.view(batch_size * self.nstack, self.in_size_extended)
    
    def _diag_third(self, S, input_x):
        batch_size = input_x.shape[0]
        output = S.view(1, -1) * input_x.view(batch_size, self.in_size_extended * self.nstack)
        # print('output after the third diag is: {}'.format(output))
        return output

    def _perm_trans(self, P, input_x):
        batch_size = input_x.shape[0]
        # output = input_x.reshape(batch_size, -1)
        output = input_x[:, P]
        # print('output after permutation is: {}'.format(output))
        return output
    
    def _hadamard_trans(self, input_x):
        H = torch.from_numpy(hadamard(self.in_size_extended)).float().cuda()
        output = torch.mm(H, input_x.transpose(1, 0))
        # output = output.transpose(1, 0)
        return output.transpose(1, 0)

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Parameters:
            input: (batch, in_size) if real 
        Return:
            output: (batch, out_size) if real 
        """  
        batch_size = input.shape[0]
        output = self.pre_process(input)
        # print('X shape after preprocess:', input.shape)
        # output = self.diag_layer1(output)
        output = self._diag_first(self.B, output)
        output = LinearFunction.apply(output)
        # output = self._hadamard_trans(output)   # output = self.hadamard_layer(output)
        # print('first hadamard:{}'.format(output.device))
        output = output.reshape(batch_size, -1)
        output = self._perm_trans(self.P, output)
        output = self._diag_secend(self.G, output)
        output = LinearFunction.apply(output)
        # output = self._hadamard_trans(output)
        output = output.reshape(batch_size, -1)
        output = self._diag_third(self.S, output) 
        # output = LinearFunction.apply(output)
        # output.cuda()
        
        return self.post_process(input, output)

    def pre_process(self, input):
        output = input.view(-1, input.size(-1))  # Reshape to (N, in_size, 1) for torch.matmul
        output = F.pad(output, (0, self.in_size_extended - self.in_size, 0, 0), "constant", value = 0)
        # print('shape of pre process tensor: {}'.format(output.shape))
        
        return output
    
    def post_process(self, input, output):
        batch = output.shape[0]
        output = output.view(batch, self.nstack * self.in_size_extended).squeeze()
        output = output[:, :self.out_size]
        # print(output.shape)
        if self.bias is not None:
            output = output + self.bias
        # print(output.shape)
        output = output.float()
        # print(output.dtype)
        
        return output

    def extra_repr(self): 
        s = 'in_size={}, out_size={}, bias={}'.format(self.in_size, self.out_size, self.bias is not None)
        return s

if __name__ == '__main__':
    input_tensor = torch.ones(4, 8).cuda()
    in_size = input_tensor.shape[1]
    out_size = 8

    ff_transform = FastfoodTransform(in_size, out_size)
    ff_transform = ff_transform.cuda()
    output = ff_transform(input_tensor)
    loss = output.sum()
    loss.backward(retain_graph=True) 
    print(output)
    print(ff_transform.B)
    print(ff_transform.G)
    print(ff_transform.S)
    print(ff_transform.B.grad)
    print(ff_transform.G.grad)
    print(ff_transform.S.grad)

    # input = (torch.randn((8, 8), requires_grad=True, dtype=torch.double))
    # input = input.cuda()
    # test = gradcheck(LinearFunction.apply, input, eps=1e-6, atol=1e-4)

