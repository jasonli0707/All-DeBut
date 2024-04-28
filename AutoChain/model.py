
import torch
import torch.nn as nn
import torch.nn.functional as F

from debut.debut_conv import DeBut_2dConv
from generate_chain import generate_debut_chains

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGG_Like(nn.Module):
    '''
    VGG-like model for MNIST dataset
    '''
    def __init__(self, debut=False, R_shapes=None, **kwargs):
        super(VGG_Like, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        if debut:
            self.conv2 = DeBut_2dConv(32, 64, 3, R_shapes = R_shapes[0], padding=1, bias=True, return_intermediates = False)
            self.conv3 = DeBut_2dConv(64, 128, 3, R_shapes = R_shapes[1], padding=1, bias=True, return_intermediates = False)
            self.conv4 = DeBut_2dConv(128, 256, 3, R_shapes = R_shapes[2], padding=1, bias=True, return_intermediates = False)
        else:
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out



if __name__ == "__main__":
    cfg = [[64, 32*3*3], [128, 64*3*3], [256, 128*3*3]]
    shrinking_level = 5
    r_shapes = generate_debut_chains(cfg, type='m', shrinking_level=shrinking_level)
    print(r_shapes)
    debut_model  = VGG_Like(debut=True, R_shapes=r_shapes)
    # print(debut_model)
    x = torch.randn(10, 1, 32, 32)
    out = debut_model(x)
    print(out.shape)
    print("shrinking level: ", shrinking_level)
    debut_params = sum(p.numel() for p in debut_model.parameters())
    print("debut num params: ", debut_params)
    model = VGG_Like()
    baseline_params = sum(p.numel() for p in model.parameters())
    print("baseline num params: ", baseline_params)
    print("MC: ", (baseline_params-debut_params)/baseline_params)
