from __future__ import print_function

import os
import argparse
import torch
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.modelnet import get_modelnet40_dataloaders
from helper.loops_pointnet import validate


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=10, help='num of workers to use')
    parser.add_argument('--model', type=str, default='pointnet_debut', choices=['pointnet', 'pointnet_debut', 'pointnet_svd', 'pointnet_butterfly'])
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'], help='dataset')
    parser.add_argument('--model_path', type=str, default=None, help='path to saved model')
    parser.add_argument('--r_shape_txt', default='./rshapes/vgg_cifar100_mono.txt', type=str, help='chains shape file')

    # PointNet
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=True, help='save data offline')

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
    if opt.dataset == 'modelnet40':
        opt.num_category = 40
    else:
        raise NotImplementedError(opt.dataset)

    data_path = os.path.join(DATA_DIR, 'modelnet40_normal_resampled')
    _, val_loader, _, test_dataset = get_modelnet40_dataloaders(data_path, opt, batch_size=opt.batch_size, num_workers=opt.num_workers, return_dataset=True)

    # model
    if opt.model == 'pointnet_debut':
        R_shapes = process_chain_file(opt.r_shape_txt)
        model = model_dict[opt.model][0](R_shapes, num_classes=opt.num_category)
    else:
        model = model_dict[opt.model][0](num_classes=opt.num_category)
    criterion = model_dict[opt.model][1]()
    # print(model)
    print("Num of parameters = " + str(sum([p.numel() for p in model.parameters()])))

    saved_dict = torch.load(opt.model_path)
    print("# epochs: ", saved_dict['epoch'])
    print("best accuracy: ", saved_dict['best_acc'].item())
    model.load_state_dict(saved_dict['model'])
    
    if torch.cuda.is_available():
        model = model.to(device)
        criterion = criterion.to(device)
        cudnn.benchmark = True

    test_class_acc, test_instance_acc, _, class_acc = validate(val_loader, model, criterion, opt, return_accuracy_per_class=True)
    # class_dict = dict([(v, k) for k, v in test_dataset.classes.items()])
    # for cat in range(opt.num_category):
    #     print(f'class {class_dict[cat]} accuracy: {class_acc[cat,2]}')
    print("class accuracy: ", test_class_acc)
    print("test_instance_acc: ", test_instance_acc)

if __name__ == '__main__':
    main()