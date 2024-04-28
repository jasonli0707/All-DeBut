from __future__ import print_function

import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders
from helper.loops import validate


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_option():
    parser = argparse.ArgumentParser('argument for testing')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=10, help='num of workers to use')
    parser.add_argument('--model', type=str, default='vgg16_bn_debut',
                        choices=['vgg16_bn', 'vgg16_bn_debut', 'vgg16_bn_butterfly', 'vgg16_bn_svd', 'vgg16_bn_fastfood', 'vgg16_bn_auto',
                                 'resnet110', 'resnet110_debut', 'resnet110_butterfly', 'resnet110_svd', 'resnet110_auto'])  
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'tinyimagenet'], help='dataset')
    parser.add_argument('--model_path', type=str, default=None, help='path to saved model')
    parser.add_argument('--r_shape_txt', default='./rshapes/vgg_cifar100_mono.txt', type=str, help='chains shape file')
    parser.add_argument('--shrinking_level', default=3, type=int, help='shrinking level for auto chain generation')
    opt = parser.parse_args()
    return opt


# Process the R chains
def process_chain_params(r_shapes):
    mat_shapes = r_shapes
    R_mats = []
    num_mat = len(r_shapes) // 5 # number of factors
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
        R_shape = process_chain_params(r_shape) # separate factors in to lists of list of integers
        R_shapes.append(R_shape)
    f.close()
    return R_shapes

def main():
    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        _, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    if opt.model.split('_')[-1] == 'debut':
        R_shapes = process_chain_file(opt.r_shape_txt)
        model = model_dict[opt.model](R_shapes, num_classes=n_cls).to(device)
    elif opt.model.split('_')[-1] == 'auto':
        model = model_dict[opt.model](num_classes=n_cls, shrinking_level=opt.shrinking_level).to(device)
    else:
        model = model_dict[opt.model](num_classes=n_cls)
        
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.to(device)
        criterion = criterion.to(device)
        cudnn.benchmark = True

    print("Num of parameters = " + str(sum([p.numel() for p in model.parameters()])))
    saved_dict = torch.load(opt.model_path)
    print("# epochs: ", saved_dict['epoch'])
    print("best accuracy: ", saved_dict['best_acc'].item())
    model.load_state_dict(saved_dict['model'])
    
    test_acc, test_acc_top5, _ = validate(val_loader, model, criterion, opt)
    print("top-1 accuracy: ", test_acc.item())
    print("top-5 accuracy: ", test_acc_top5.item())

    print(test_acc.item()/100, test_acc_top5.item()/100)

if __name__ == '__main__':
    main()
