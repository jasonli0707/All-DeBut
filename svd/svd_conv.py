from audioop import bias
import math
import torch
from torch import conv2d, nn
import torch.nn.functional as F
import numpy as np

class SVD_2dConv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0, dilation = 1, bias=True, compression_rate=0.5):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.in_size = self.input_channel * self.kernel_size[0] * self.kernel_size[1]
        self.out_size = self.output_channel
        conv = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding, dilation, bias=False)
        conv_weight = conv.weight
        self.weight = conv_weight.view(conv_weight.size(0), -1).t() # c_o, k^2*c_i
        U, SVs, Vt = np.linalg.svd(self.weight.detach().numpy())
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_channel))
            num_params_bias = self.bias.numel()
        else:
            self.register_parameter('bias', None)
            num_params_bias = 0
        self.reset_parameters()

 
        num_params_conv = self.weight.numel()
        num_params = num_params_conv + num_params_bias
        target_params = num_params * compression_rate

        num_SVs = len(SVs)
        num_selected_SVs = 0
        for i in range(num_SVs):
            num_params_svd = U[:, :i].size + SVs[:i].size + Vt[:i].size + num_params_bias
            if num_params_svd >= target_params:
                num_selected_SVs = i 
                break

        # print(f'selected SVs: {num_selected_SVs}/{num_SVs}')

        self.U_cut = torch.nn.Parameter(torch.tensor(U[:, :num_selected_SVs]))  # [C_o, num_selected_SVs]
        self.S_cut = torch.nn.Parameter(torch.tensor(np.diag(SVs[:num_selected_SVs]))) # [num_selected_SVs, num_selected_SVs]
        self.V_cut = torch.nn.Parameter(torch.tensor(Vt[:num_selected_SVs])) # [C_i, num_selected_SVs]
        # print('The shapes of U_cut, S_cut, and V_cut are: {}, {}, and {}.'.format(self.U_cut.shape, self.S_cut.shape, self.V_cut.shape))


    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        batch_size, channels, h_in, w_in = input.shape
        inp_unf = F.unfold(input, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation)
        h_out = (h_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (w_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        weight_cut = self.U_cut @ self.S_cut @ self.V_cut
        out_unf = inp_unf.transpose(1, 2).matmul(weight_cut)
        if self.bias is not None:
            out_unf += self.bias
        out_unf = out_unf.transpose(1, 2)
        return out_unf.view(batch_size, self.output_channel, h_out, w_out)


class SVD_1dConv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0, dilation = 1, bias=True, compression_rate=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.weight = torch.Tensor(output_channel, input_channel, kernel_size)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_channel))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # SVD 
        # only valid for kernel size = 1 
        U, SVs, Vt = np.linalg.svd(self.weight.squeeze())
        num_params_bias = self.bias.numel()
        num_params_conv = self.weight.numel()
        total_params = num_params_conv + num_params_bias
        # use compression rate to decide the number of selected SVs
        num_SVs = len(SVs)
        num_selected_SVs = 0
        target_params = total_params * compression_rate
        for i in range(num_SVs):
            num_params_svd = U[:, :i].size + SVs[:i].size + Vt[:i].size + num_params_bias
            if num_params_svd >= target_params:
                num_selected_SVs = i
                break

        # print(f'selected SVs: {num_selected_SVs}/{num_SVs}')
        self.U_cut = torch.nn.Parameter(torch.tensor(U[:, :num_selected_SVs]))  # [C_o, num_selected_SVs]
        self.S_cut = torch.nn.Parameter(torch.tensor(np.diag(SVs[:num_selected_SVs]))) # [num_selected_SVs, num_selected_SVs]
        self.V_cut = torch.nn.Parameter(torch.tensor(Vt[:num_selected_SVs])) # [C_i, num_selected_SVs]
        # print('The shapes of U_cut, S_cut, and V_cut are: {}, {}, and {}.'.format(self.U_cut.shape, self.S_cut.shape, self.V_cut.shape))


    def reset_parameters(self):
        """Initialize bias the same way as torch.nn._ConvNd."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight_cut = self.U_cut @ self.S_cut @ self.V_cut
        return F.conv1d(input, weight_cut.unsqueeze(-1), self.bias, self.stride,
                        self.padding, self.dilation)



if __name__ == '__main__':
    x = torch.randn(1, 3, 16, 16)
    conv = nn.Conv2d(3, 3, 3 , padding=1)
    print(sum([p.numel()for p in conv.parameters()]))
    svd_conv = SVD_2dConv(3, 3, 3 , padding=1, compression_rate=0.5)
    for n, p in svd_conv.named_parameters():
        print(n, p.numel())
    print(sum([p.numel()for p in svd_conv.parameters()]))
    actual = conv(x)
    result = svd_conv(x)
    print(actual.shape)
    print(result.shape)
