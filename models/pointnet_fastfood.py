from random import randint
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from fastfood.fastfood_conv import Fastfood_1dConv
from fastfood.fastfood_transform import FastfoodTransform

device = 'cuda' if torch.cuda.is_available() else 'cpu'

__all__ = ['pointnet_fastfood', 'pointnet_fastfood_loss']

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = Fastfood_1dConv(channel, 64, 1)
        self.conv2 = Fastfood_1dConv(64, 128, 1)
        self.conv3 = Fastfood_1dConv(128, 1024, 1)
        self.fc1 = FastfoodTransform(1024, 512)
        self.fc2 = FastfoodTransform(512, 256)
        self.fc3 = FastfoodTransform(256, 9)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, channel=64):
        super(STNkd, self).__init__()
        self.conv1 = Fastfood_1dConv(channel, 64, 1)
        self.conv2 = Fastfood_1dConv(64, 128, 1)
        self.conv3 = Fastfood_1dConv(128, 1024, 1)
        self.fc1 = FastfoodTransform(1024, 512)
        self.fc2 = FastfoodTransform(512, 256)
        self.fc3 = FastfoodTransform(256, channel * channel)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.channel = channel

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.channel).flatten().astype(np.float32))).view(1, self.channel * self.channel).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.channel, self.channel)
        return x

class PointNet_Fastfood(nn.Module):
    def __init__(self, classes=40, normal_channel=False, feature_transform=True):
        super(PointNet_Fastfood, self).__init__() 
        self.feature_transform = feature_transform
        if normal_channel:
            self.channel = 6
        else:
            self.channel = 3
        self.stn = STN3d(self.channel)
        self.block0 = self._make_block_0(channel=self.channel)
        if self.feature_transform:
            print('Using feature transform...')
            self.fstn = STNkd(channel=64)
        self.block1 = self._make_block_1()
        self.block2 = self._make_block_2()        
        self.classifier = FastfoodTransform(256, classes)
        
    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.stn)
        feat_m.append(self.block0)
        if self.feature_transform:
            feat_m.append(self.fstn)
        feat_m.append(self.block1)
        feat_m.append(self.block2)
        return feat_m

    def forward(self, x, is_feat=False):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        f0 = x
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1) # B, C_in, L_in
        x = self.block0(x)
        f1 = x
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
        f2 = x
        x = self.block1(x)
        f3 = x
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.block2(x)
        f4 = x
        x = self.classifier(x) 
        if is_feat:
            return [f0, f1, f2, f3, f4], x, trans_feat
        else:
            return x, trans_feat

    @staticmethod
    def _make_block_0(channel=3):
        conv1 = Fastfood_1dConv(channel, 64, 1)
        bn1 = nn.BatchNorm1d(64)
        relu = nn.ReLU()
        layers = [conv1, bn1, relu]
        return nn.Sequential(*layers)

    @staticmethod
    def _make_block_1():
        conv2 = Fastfood_1dConv(64, 128, 1)
        conv3 = Fastfood_1dConv(128, 1024, 1)
        bn2 = nn.BatchNorm1d(128)
        bn3 = nn.BatchNorm1d(1024)
        relu = nn.ReLU()
        layers = [conv2, bn2, relu, conv3, bn3]
        return nn.Sequential(*layers)
        
    @staticmethod
    def _make_block_2():
        fc1 = FastfoodTransform(1024, 512)
        fc2 = FastfoodTransform(512, 256)
        dropout = nn.Dropout(p=0.4)
        bn1 = nn.BatchNorm1d(512)
        bn2 = nn.BatchNorm1d(256)
        relu = nn.ReLU()
        layers = [fc1, bn1, relu, fc2, dropout, bn2, relu]
        return nn.Sequential(*layers)

class pointnet_fastfood_loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(pointnet_fastfood_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, logits, target, trans_feat, feature_transform=True):
        loss = F.cross_entropy(logits, target)
        if feature_transform:
            mat_diff_loss = feature_transform_regularizer(trans_feat)
            total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
            return total_loss
        return loss

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

def pointnet_fastfood(num_classes=40, **kwargs):
    model = PointNet_Fastfood(classes=num_classes, **kwargs)
    return model

if __name__ == "__main__":
    model = pointnet_fastfood().to(device)
    print(model)
    print("Num of parameters = " + str(sum([p.numel() for p in model.parameters()])))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())

    x = torch.randn(10, 3, 100).to(device)
    features, output, trans_feat = model(x, is_feat=True)
    print(output.shape)
    