import torch.nn as nn
import torch.nn.functional as F
import math
from debut.debut import DeBut
from debut.debut_conv import DeBut_2dConv
from AutoChain.generate_chain import generate_debut_chains

__all__ = ['resnet110_auto']


def debut_conv3x3(in_planes, out_planes, stride=1, R_shapes=None):
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    return DeBut_2dConv(in_planes, out_planes, 3, stride=stride, padding=1, bias=False, R_shapes=R_shapes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False, R_shapes_1=None, R_shapes_2=None):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = debut_conv3x3(inplanes, planes, stride, R_shapes=R_shapes_1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = debut_conv3x3(planes, planes, R_shapes=R_shapes_2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet110_Auto(nn.Module):

    def __init__(self, depth, num_filters, num_classes=100, shrinking_level=3):
        super(ResNet110_Auto, self).__init__()
        """
        depth: 110
        num_filters: [16, 16, 32, 64]
        """
        assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
        n = (depth - 2) // 6 # 18
        block = BasicBlock

        # auto chain generation
        self.shrinking_level = shrinking_level
        cfg = [[16, 144], [32, 144], [32, 288], [64, 288], [64, 576]] 
        R_shapes = generate_debut_chains(cfg, shrinking_level=self.shrinking_level)

        self.inplanes = num_filters[0] # 16
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer_1(block, n, R_shapes)
        self.layer2 = self._make_layer_2(block, n, R_shapes)
        self.layer3 = self._make_layer_3(block, n, R_shapes)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_1(self, block, blocks, R_shapes):
        layers = list([])
        # [16, 27]
        layers.append(block(16, 16, 1, is_last=False, R_shapes_1=R_shapes[0], R_shapes_2=R_shapes[0])) 
        for i in range(1, blocks):
            # [16, 27]
            layers.append(block(16, 16, is_last=(i==blocks-1), R_shapes_1=R_shapes[0], R_shapes_2=R_shapes[0]))
        return nn.Sequential(*layers)

    def _make_layer_2(self, block, blocks, R_shapes):
        # [32, 16]
        downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32)
        )
        layers = list([])
        # [32, 144]
        layers.append(block(16, 32, 2, downsample, is_last=False, R_shapes_1=R_shapes[1], R_shapes_2=R_shapes[2])) 
        for i in range(1, blocks):
            # [32, 288]
            layers.append(block(32, 32, is_last=(i==blocks-1), R_shapes_1=R_shapes[2], R_shapes_2=R_shapes[2]))
        return nn.Sequential(*layers)

    def _make_layer_3(self, block, blocks, R_shapes):
        # [64, 32]
        downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64)
        )
        layers = list([])
        #[64, 288]
        layers.append(block(32, 64, 2, downsample, is_last=False, R_shapes_1=R_shapes[3], R_shapes_2=R_shapes[4])) 
        for i in range(1, blocks):
            # [64, 576]
            layers.append(block(64, 64, is_last=(i==blocks-1), R_shapes_1=R_shapes[4], R_shapes_2=R_shapes[4]))
        return nn.Sequential(*layers)


    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3]

    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        f0 = x

        x, f1_pre = self.layer1(x)  # 32x32
        f1 = x
        x, f2_pre = self.layer2(x)  # 16x16
        f2 = x
        x, f3_pre = self.layer3(x)  # 8x8
        f3 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f4 = x
        x = self.fc(x)

        if is_feat:
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4], x
            else:
                return [f0, f1, f2, f3, f4], x
        else:
            return x

def resnet110_auto(num_classes=100, shrinking_level=3):
    return ResNet110_Auto(110, [16, 16, 32, 64], num_classes=num_classes, shrinking_level=shrinking_level)


if __name__ == '__main__':
    import torch
    x = torch.randn(2, 3, 32, 32) 
    net = resnet110_auto(num_classes=100, shrinking_level=5)
    feats, logit = net(x, is_feat=True, preact=True)
    print(logit.shape)
    print("num of params: %d" % (sum([p.numel() for p in net.parameters()])))
    # for n, p in net.named_parameters():
    #     print('name: ', n)
    #     print('parameters: ', p.numel())