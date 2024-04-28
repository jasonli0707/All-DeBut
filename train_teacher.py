from __future__ import print_function

import os
import argparse
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate
from helper.loops import train_vanilla as train, validate


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_option():


    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=10, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['vgg16_bn', 'vgg16_bn_debut', 'vgg16_bn_butterfly', 'vgg16_bn_svd', 'vgg16_bn_fastfood', 'vgg16_bn_auto',
                                 'resnet110', 'resnet110_debut', 'resnet110_butterfly', 'resnet110_svd', 'resnet110_auto'])  
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')
    
    # DeBut model definition
    parser.add_argument('--r_shape_txt', default=None, type=str, help='chains shape file')

    opt = parser.parse_args()
    
    # set the path according to the environment

    opt.model_path = './save/models'
    opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_epoch_{}_batch_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.epochs, opt.batch_size, opt.learning_rate, opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

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
    best_acc = 0

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    if opt.model == 'vgg16_bn_debut' or opt.model == 'resnet110_debut':
        R_shapes = process_chain_file(opt.r_shape_txt)
        model = model_dict[opt.model](R_shapes, num_classes=n_cls).to(device)
    else:
        model = model_dict[opt.model](num_classes=n_cls)
        

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.to(device)
        criterion = criterion.to(device)
        cudnn.benchmark = True

    print(model)
    print("Num of parameters = " + str(sum(param.numel() for param in model.parameters())))

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
