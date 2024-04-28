from __future__ import print_function

import os
import argparse
import time

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.modelnet import get_modelnet40_dataloaders
from helper.loops_pointnet import train_vanilla as train, validate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=10, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

    # dataset
    parser.add_argument('--model', type=str, default='pointnet', choices=['pointnet'])
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'], help='dataset')
    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    # PointNet
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=True, help='save data offline')

    opt = parser.parse_args()
    

    # set the path according to the environment
    opt.model_path = './save/models'
    opt.tb_path = './save/tensorboard'

    opt.model_name = '{}_{}_epoch_{}_batch_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.epochs, opt.batch_size, opt.learning_rate, opt.weight_decay, opt.trial)

    print("model name: ", opt.model_name)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    best_acc = 0
    opt = parse_option()

    # dataloader
    if opt.dataset == 'modelnet40':
        opt.num_category = 40
    else:
        raise NotImplementedError(opt.dataset)

    data_path = os.path.join(DATA_DIR, 'modelnet40_normal_resampled')
    train_loader, val_loader, n_data = get_modelnet40_dataloaders(data_path, opt, batch_size=opt.batch_size, num_workers=opt.num_workers)

    # model
    if opt.model == 'pointnet':
        model = model_dict[opt.model][0](num_classes=opt.num_category)
        criterion = model_dict[opt.model][1]()
        print(model)
        print("Num of parameters = " + str(sum([p.numel() for p in model.parameters()])))
    else:
        raise NotImplementedError(opt.model)
  
    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=opt.weight_decay
    )

    if torch.cuda.is_available():
        model = model.to(device)
        criterion = criterion.to(device)
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    for epoch in range(1, opt.epochs + 1):
        print("==> training...")
        time1 = time.time()
        scheduler.step()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_class_acc, test_instance_acc, test_loss = validate(val_loader, model, criterion, opt)

        logger.log_value('test_class_acc', test_class_acc, epoch)
        logger.log_value('test_instance_acc', test_instance_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_instance_acc > best_acc:
            best_acc = test_instance_acc
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
                'instance_acc': test_instance_acc,
                'class_acc': test_class_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('Best Instance Accuracy: ', best_acc)


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