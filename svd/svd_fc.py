import torch
import numpy as np
import torch.nn as nn
import math

# define the FC layer decomposed by SVD
class SVD_Linear(nn.Module):
    """
    Use Singular Value Decomposition (SVD, A=USV^T) to decompose the fully-connected (FC) layer.

    Init params:
        - fc_nn_module: a selected FC layer, which belongs to the nn.Module class.
        - compression_rate: a float number, which decides the number of selected SVs and the size of U,S and V.
    """

    def __init__(self, fc_nn_module, compression_rate=0.8, bias=True):
        super(SVD_Linear, self).__init__()

        fc_weight = fc_nn_module.weight.data
        fc_bias = fc_nn_module.bias.data
        num_params_fc = fc_weight.numel() + fc_bias.numel()

        self.in_size = fc_nn_module.weight.size(1)
        self.out_size = fc_nn_module.weight.size(0)
        U, SVs, Vt = np.linalg.svd(fc_weight)

        # use compression rate to decide the number of selected SVs
        num_SVs = len(SVs)
        target_params = num_params_fc * compression_rate
        for i in range(num_SVs):
            num_params_svd = U[:, :i].size + SVs[:i].size + Vt[:i].size + fc_bias.numel()
            if num_params_svd >= target_params:
                num_selected_SVs = i 
                break
        
        # print(f'selected SVs: {num_selected_SVs}/{num_SVs}')

        self.U_cut = torch.nn.Parameter(torch.tensor(U[:, :num_selected_SVs]))  # [C_o, num_selected_SVs]
        self.S_cut = torch.nn.Parameter(torch.tensor(np.diag(SVs[:num_selected_SVs])))  # [num_selected_SVs, num_selected_SVs]
        self.V_cut = torch.nn.Parameter(torch.tensor(Vt[:num_selected_SVs].T))  # [C_i, num_selected_SVs].T
        if bias:
            bias_shape = (self.out_size, ) 
            self.bias = nn.Parameter(torch.Tensor(*bias_shape))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # print('The shapes of U_cut, S_cut, and V_cut are: {}, {}, and {}.'.format(self.U_cut.shape, self.S_cut.shape, self.V_cut.shape))

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        The weight matrix (denoted as A) of a FC layer is of size [C_o, C_i], and the
        size of the input (denoted as x) before the FC layer is [Batch size, C_i].

        The conventional way to get the output of the FC layer is output = x * A.T. After
        implying SVD on the FC weight matrix, we get the output of the FC layer by three
        steps: 1) x = x * V_cut, 2) x = x * S_cut.T, 3) x = x * U_cut.T.
        """
        x = torch.matmul(x, self.V_cut)
        x = torch.matmul(x, self.S_cut.T)
        x = torch.matmul(x, self.U_cut.T)
        out = x + self.bias
        return out

if __name__ == '__main__':
    fc = nn.Linear(256, 4096)
    fc_params = sum([p.numel()for p in fc.parameters()])
    print("original:")
    print(fc_params)
 
    svd = SVD_Linear(fc, compression_rate=0.2)
    svd_params = sum([p.numel()for p in svd.parameters()])
    svd_compression_rate = svd_params/fc_params
    print('svd:')
    print(svd_params)
    print('compression rate: ', svd_params/fc_params)

    x = torch.randn(100, 256)
    out = svd(x)
    print(out.shape)