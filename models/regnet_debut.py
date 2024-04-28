'''RegNet in PyTorch.
Paper: "Designing Network Design Spaces".
Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from debut.debut import DeBut
from debut.debut_conv import DeBut_2dConv


__all__ = ['regnetx400mf_debut', 'regnetx200mf_debut']


class SE(nn.Module):
    '''Squeeze-and-Excitation block.'''

    def __init__(self, in_planes, se_planes):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_planes, se_planes, kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_planes, in_planes, kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = F.relu(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    def __init__(self, w_in, w_out, stride, bottleneck_ratio, se_ratio,
                R_shapes_1=None, R_shapes_2=None, R_shapes_3=None, R_shapes_4=None):
        super(Block, self).__init__()
        # 1x1
        w_b = int(round(w_out * bottleneck_ratio))
        self.conv1 = DeBut_2dConv(w_in, w_b, kernel_size=1, bias=False, R_shapes=R_shapes_1)
        self.bn1 = nn.BatchNorm2d(w_b)
        # 3x3
        self.conv2 = DeBut_2dConv(w_b, w_b, kernel_size=3, stride=stride, padding=1, bias=False, R_shapes=R_shapes_2)
        self.bn2 = nn.BatchNorm2d(w_b)
        # se
        self.with_se = se_ratio > 0
        if self.with_se:
            w_se = int(round(w_in * se_ratio))
            self.se = SE(w_b, w_se)
        # 1x1
        self.conv3 = DeBut_2dConv(w_b, w_out, kernel_size=1, bias=False, R_shapes=R_shapes_3)
        self.bn3 = nn.BatchNorm2d(w_out)

        self.shortcut = nn.Sequential()
        if stride != 1 or w_in != w_out:
            self.shortcut = nn.Sequential(
                DeBut_2dConv(w_in, w_out, kernel_size=1, stride=stride, bias=False, R_shapes=R_shapes_4),
                nn.BatchNorm2d(w_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.with_se:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RegNet_DeBut(nn.Module):
    def __init__(self, cfg, num_classes=10, R_shapes=None):
        super(RegNet_DeBut, self).__init__()
        self.cfg = cfg
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer1(R_shapes)
        self.layer2 = self._make_layer2(R_shapes)
        self.layer3 = self._make_layer3(R_shapes)
        self.layer4 = self._make_layer4(R_shapes)
        self.linear = nn.Linear(self.cfg['widths'][-1], num_classes)

    def _make_layer1(self, R_shapes):
        depth = self.cfg['depths'][0]
        width = self.cfg['widths'][0]
        stride = self.cfg['strides'][0]
        bottleneck_ratio = self.cfg['bottleneck_ratio']
        se_ratio = self.cfg['se_ratio']

        layers = []
        for i in range(depth):
            s = stride if i == 0 else 1
            layers.append(Block(self.in_planes, width, s, bottleneck_ratio, se_ratio, 
                                R_shapes_1=R_shapes[0], R_shapes_2=R_shapes[1], R_shapes_3=R_shapes[2], R_shapes_4=R_shapes[0]))
            self.in_planes = width
        return nn.Sequential(*layers)

    def _make_layer2(self, R_shapes):
        depth = self.cfg['depths'][1]
        width = self.cfg['widths'][1]
        stride = self.cfg['strides'][1]
        bottleneck_ratio = self.cfg['bottleneck_ratio']
        se_ratio = self.cfg['se_ratio']

        layers = []
        for i in range(depth):
            s = stride if i == 0 else 1
            if i == 0: # block1
                layers.append(Block(self.in_planes, width, s, bottleneck_ratio, se_ratio, 
                                R_shapes_1=R_shapes[5], R_shapes_2=R_shapes[3], R_shapes_3=R_shapes[4], R_shapes_4=R_shapes[5]))
            else: # block 2
                layers.append(Block(self.in_planes, width, s, bottleneck_ratio, se_ratio, 
                    R_shapes_1=R_shapes[4], R_shapes_2=R_shapes[3], R_shapes_3=R_shapes[4], R_shapes_4=None))
            self.in_planes = width
        return nn.Sequential(*layers)

    def _make_layer3(self, R_shapes):
        depth = self.cfg['depths'][2]
        width = self.cfg['widths'][2]
        stride = self.cfg['strides'][2]
        bottleneck_ratio = self.cfg['bottleneck_ratio']
        se_ratio = self.cfg['se_ratio']

        layers = []
        for i in range(depth):
            s = stride if i == 0 else 1
            if i == 0: # block 1
                layers.append(Block(self.in_planes, width, s, bottleneck_ratio, se_ratio, 
                                R_shapes_1=R_shapes[6], R_shapes_2=R_shapes[7], R_shapes_3=R_shapes[8], R_shapes_4=R_shapes[6]))
            else: # block 2-7
                layers.append(Block(self.in_planes, width, s, bottleneck_ratio, se_ratio, 
                    R_shapes_1=R_shapes[8], R_shapes_2=R_shapes[7], R_shapes_3=R_shapes[8], R_shapes_4=None))
            self.in_planes = width
        return nn.Sequential(*layers)

    def _make_layer4(self, R_shapes):
        depth = self.cfg['depths'][3]
        width = self.cfg['widths'][3]
        stride = self.cfg['strides'][3]
        bottleneck_ratio = self.cfg['bottleneck_ratio']
        se_ratio = self.cfg['se_ratio']

        layers = []
        for i in range(depth):
            s = stride if i == 0 else 1
            if i == 0: # block 1
                layers.append(Block(self.in_planes, width, s, bottleneck_ratio, se_ratio, 
                                R_shapes_1=R_shapes[9], R_shapes_2=R_shapes[10], R_shapes_3=R_shapes[11], R_shapes_4=R_shapes[9]))
            else: # block 2-12
                layers.append(Block(self.in_planes, width, s, bottleneck_ratio, se_ratio, 
                    R_shapes_1=R_shapes[11], R_shapes_2=R_shapes[10], R_shapes_3=R_shapes[11], R_shapes_4=None))
            self.in_planes = width
        return nn.Sequential(*layers)
    
    def forward(self, x, is_feat=False, preact=False):
        out = F.relu(self.bn1(self.conv1(x)))
        f0 = out
        out = self.layer1(out)
        f1 = out
        out = self.layer2(out)
        f2 = out 
        out = self.layer3(out)
        f3 = out
        out = self.layer4(out)
        f4 = out
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.linear(out)
        if is_feat:
            return [f0, f1, f2, f3, f4, f5], out
        else:
            return out


def regnetx200mf_debut(R_shapes, num_classes=100):
    cfg = {
        'depths': [1, 1, 4, 7],
        'widths': [24, 56, 152, 368],
        'strides': [1, 1, 2, 2],
        'bottleneck_ratio': 1,
        'se_ratio': 0,
    }
    return RegNet_DeBut(cfg, num_classes=num_classes, R_shapes=R_shapes)


def regnetx400mf_debut(R_shapes, num_classes=100):
    cfg = {
        'depths': [1, 2, 7, 12],
        'widths': [32, 64, 160, 384],
        'strides': [1, 1, 2, 2],
        'bottleneck_ratio': 1,
        'se_ratio': 0,
    }
    return RegNet_DeBut(cfg, num_classes=num_classes, R_shapes=R_shapes)
