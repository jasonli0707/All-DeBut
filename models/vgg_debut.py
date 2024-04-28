import torch.nn as nn
import torch.nn.functional as F
import math
import torch

from debut.debut_conv import DeBut_2dConv
from debut.debut import DeBut

device = 'cuda' if torch.cuda.is_available() else 'cpu'

__all__ = ['vgg16_bn_debut']

class VGG_DeBut(nn.Module):

    def __init__(self, R_shapes, num_classes=100):
        super(VGG_DeBut, self).__init__()
        
        self.R_shapes = R_shapes
    
        self.block0 = self._make_layers_0(self.R_shapes)
        self.block1 = self._make_layers_1(self.R_shapes)
        self.block2 = self._make_layers_2(self.R_shapes)
        self.block3 = self._make_layers_3(self.R_shapes)
        self.block4 = self._make_layers_4(self.R_shapes)

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.classifier = nn.Linear(512, num_classes)
        self.classifier = DeBut(512, num_classes, R_shapes=R_shapes[-1])
        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([]).to(device)
        feat_m.append(self.block0).to(device)
        feat_m.append(self.pool0).to(device)
        feat_m.append(self.block1).to(device)
        feat_m.append(self.pool1).to(device)
        feat_m.append(self.block2).to(device)
        feat_m.append(self.pool2).to(device)
        feat_m.append(self.block3).to(device)
        feat_m.append(self.pool3).to(device)
        feat_m.append(self.block4).to(device)
        feat_m.append(self.pool4).to(device)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, is_feat=False, preact=False):
        h = x.shape[2] # x.shape: (b, c_i, h, w)
        x = F.relu(self.block0(x.to(device)))
        f0 = x.to(device)
        x = self.pool0(x).to(device)
        x = self.block1(x).to(device)
        f1_pre = x.to(device)
        x = F.relu(x).to(device)
        f1 = x.to(device)
        x = self.pool1(x).to(device)
        x = self.block2(x).to(device)
        f2_pre = x.to(device)
        x = F.relu(x).to(device)
        f2 = x.to(device)
        x = self.pool2(x).to(device)
        x = self.block3(x).to(device)
        f3_pre = x.to(device)
        x = F.relu(x).to(device)
        f3 = x.to(device)
        if h == 64:
            x = self.pool3(x).to(device)
        x = self.block4(x).to(device)
        f4_pre = x.to(device)
        x = F.relu(x).to(device)
        f4 = x.to(device)
        x = self.pool4(x).to(device)
        x = x.view(x.size(0), -1).to(device)
        f5 = x.to(device)
        x = self.classifier(x).to(device)

        if is_feat:
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x
            else:
                return [f0, f1, f2, f3, f4, f5], x
        else:
            return x

    # [64, 64]
    @staticmethod
    def _make_layers_0(R_shapes):
        layers = []
        conv2d_1 = DeBut_2dConv(3, 64, 3, padding=1, R_shapes=R_shapes[0], bias=True, return_intermediates=False)
        conv2d_2 = DeBut_2dConv(64, 64, 3, padding=1, R_shapes=R_shapes[1], bias=True, return_intermediates=False)
        layers = [conv2d_1, nn.BatchNorm2d(64), nn.ReLU(inplace=True), conv2d_2, nn.BatchNorm2d(64)]
        return nn.Sequential(*layers)
    
    # [128, 128]
    @staticmethod
    def _make_layers_1(R_shapes):
        layers = []
        conv2d_1 = DeBut_2dConv(64, 128, 3, padding=1, R_shapes=R_shapes[2], bias=True, return_intermediates=False)
        conv2d_2 = DeBut_2dConv(128, 128, 3, padding=1, R_shapes=R_shapes[3], bias=True, return_intermediates=False)
        layers = [conv2d_1, nn.BatchNorm2d(128), nn.ReLU(inplace=True), conv2d_2, nn.BatchNorm2d(128)]
        return nn.Sequential(*layers)
    
    # [256, 256, 256]
    @staticmethod
    def _make_layers_2(R_shapes):
        layers = []
        conv2d_1 = DeBut_2dConv(128, 256, 3, padding=1, R_shapes=R_shapes[4], bias=True, return_intermediates=False)
        conv2d_2 = DeBut_2dConv(256, 256, 3, padding=1, R_shapes=R_shapes[5], bias=True, return_intermediates=False)
        conv2d_3 = DeBut_2dConv(256, 256, 3, padding=1, R_shapes=R_shapes[6], bias=True, return_intermediates=False)
        layers = [conv2d_1, nn.BatchNorm2d(256), nn.ReLU(inplace=True), conv2d_2, nn.BatchNorm2d(256), nn.ReLU(inplace=True), conv2d_3, nn.BatchNorm2d(256)]
        return nn.Sequential(*layers)
    
    # [512, 512, 512]
    @staticmethod
    def _make_layers_3(R_shapes):
        layers = []
        conv2d_1 = DeBut_2dConv(256, 512, 3, padding=1, R_shapes=R_shapes[7], bias=True, return_intermediates=False)
        conv2d_2 = DeBut_2dConv(512, 512, 3, padding=1, R_shapes=R_shapes[8], bias=True, return_intermediates=False)
        conv2d_3 = DeBut_2dConv(512, 512, 3, padding=1, R_shapes=R_shapes[9], bias=True, return_intermediates=False)
        layers = [conv2d_1, nn.BatchNorm2d(512), nn.ReLU(inplace=True), conv2d_2, nn.BatchNorm2d(512), nn.ReLU(inplace=True), conv2d_3, nn.BatchNorm2d(512)]
        return nn.Sequential(*layers)
    
    # [512, 512, 512]
    @staticmethod
    def _make_layers_4(R_shapes):
        layers = []
        conv2d_1 = DeBut_2dConv(512, 512, 3, padding=1, R_shapes=R_shapes[10], bias=True, return_intermediates=False)
        conv2d_2 = DeBut_2dConv(512, 512, 3, padding=1, R_shapes=R_shapes[11], bias=True, return_intermediates=False)
        conv2d_3 = DeBut_2dConv(512, 512, 3, padding=1, R_shapes=R_shapes[12], bias=True, return_intermediates=False)
        layers = [conv2d_1, nn.BatchNorm2d(512), nn.ReLU(inplace=True), conv2d_2, nn.BatchNorm2d(512), nn.ReLU(inplace=True), conv2d_3, nn.BatchNorm2d(512)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def vgg16_bn_debut(R_shapes, num_classes=100):
    model = VGG_DeBut(R_shapes, num_classes=num_classes)
    return model

# Process the R chains
def process_chain_params(r_shapes):
    mat_shapes = r_shapes
    R_mats = []
    num_mat = len(r_shapes) // 5
    for i in range(num_mat):
        start_i, end_i = i * 5, (i + 1) * 5
        R_mat = mat_shapes[start_i: end_i]
        R_mats.append(R_mat)
    return R_mats


def process_chain_file(r_shapes_file):
    R_shapes = []
    f = open(r_shapes_file, 'r')
    while True:
        line = f.readline()
        if len(line) <= 1:
            break
        temp = line.strip('\n').split(' ')
        r_shape = [int(a) for a in temp]
        R_shape = process_chain_params(r_shape)
        R_shapes.append(R_shape)
    f.close()
    return R_shapes

if __name__ == '__main__':
    R_shapes = process_chain_file('./rshapes/vgg_cifar100_mono.txt')
    model = vgg16_bn_debut(R_shapes, 100)
    print(model)
    print('# vgg16_bn_debut parameters:', sum(param.numel() for param in model.parameters()))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())
    model = model.to(device)
    input = torch.randn(1, 3, 32, 32)
    output = model(input.to(device))
    print(output.shape)
