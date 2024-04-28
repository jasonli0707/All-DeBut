from __future__ import print_function, division

import sys
import torch
import numpy as np
from tqdm import tqdm

import provider

def _augment_batch_data(batch_data):
    jittered_data = provider.random_scale_point_cloud(batch_data[:,:,0:3])
    jittered_data = provider.shift_point_cloud(jittered_data)
    return jittered_data

def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()
    mean_acc = []
    mean_loss = []
    
    for idx, (points, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        points = points.data.numpy()
        points = _augment_batch_data(points)
        points = torch.Tensor(points)
        points = points.transpose(2, 1)

        if torch.cuda.is_available():
            input = points.cuda()
            target = target.cuda()

        # ===================forward=====================
        logits, trans_feat = model(input)
        loss = criterion(logits, target.long(), trans_feat)

        pred_choice = logits.data.max(1)[1] # predicted classes: (batch_size)
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_acc.append(correct.item() / float(points.size()[0]))
        mean_loss.append(loss.item())
        train_instance_acc = np.mean(mean_acc)
        train_loss = np.mean(mean_loss)
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Train Loss {loss:.4f} ({loss_avg:.4f})\t'
                  'Train Acc {train_acc:.3f}) ({train_acc_avg:.3f})'.format(
                   epoch, idx, len(train_loader), loss=mean_loss[idx], loss_avg=train_loss, train_acc=mean_acc[idx], train_acc_avg=train_instance_acc))
            sys.stdout.flush()

    train_loss = np.mean(mean_loss)
    train_instance_acc = np.mean(mean_acc)
    print(' * Train Accuracy {train_acc:.3f}'.format(train_acc=train_instance_acc))
    return train_instance_acc, train_loss


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    mean_acc = []
    mean_loss = []

    
    for idx, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data

        input = input.data.numpy()
        input = provider.random_point_dropout(input)
        input = _augment_batch_data(input)
        input = torch.Tensor(input)
        input = input.transpose(2, 1)
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        # preact = False
        # if opt.distill in ['abound']:
        #     preact = True
        feat_s, logit_s, trans_feat_s = model_s(input, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t, _ = model_t(input, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target.long(), trans_feat_s)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        pred_choice = logit_s.data.max(1)[1] # predicted classes: (batch_size)
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_acc.append(correct.item() / float(input.size()[0]))
        mean_loss.append(loss.item())
        train_loss = np.mean(mean_loss)
        train_instance_acc = np.mean(mean_acc)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Train Loss {loss:.4f} ({loss_avg:.4f})\t'
                  'Train Acc {train_acc:.3f} ({train_acc_avg:.3f})'.format(
                   epoch, idx, len(train_loader), loss=mean_loss[idx], loss_avg=train_loss, train_acc=mean_acc[idx], train_acc_avg=train_instance_acc))
            sys.stdout.flush()

    train_loss = np.mean(mean_loss)
    train_instance_acc = np.mean(mean_acc)
    print(' * Train Accuracy {train_acc:.3f}'.format(train_acc=train_instance_acc))
    return train_instance_acc, train_loss


def validate(val_loader, model, criterion, opt, return_accuracy_per_class=False):
    """validation"""
    # switch to evaluate mode
    model.eval()
    mean_loss = []
    mean_acc = []
    class_acc = np.zeros((opt.num_category, 3))
    with torch.no_grad():
        for idx, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):

            input = input.transpose(2, 1)
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            logits, trans_feat = model(input)
            loss = criterion(logits, target.long(), trans_feat)
            pred_choice = logits.data.max(1)[1]

            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(input[target == cat].size()[0])
                class_acc[cat, 1] += 1
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_acc.append(correct.item()/float(input.size(0)))
            mean_loss.append(loss.item())

            test_instance_acc = np.mean(mean_acc)
            test_loss = np.mean(mean_loss)

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss:.4f} ({loss_avg:.4f})\t'
                      'Instance Accuracy {test_accuracy:.3f}'.format(
                       idx, len(val_loader), loss=mean_loss[idx], loss_avg=test_loss,
                       test_accuracy=test_instance_acc))

        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        test_loss = np.mean(mean_loss)
        test_instance_acc = test_instance_acc
        test_class_acc = np.mean(class_acc[:, 2])
        print(' * Class Accuracy {class_accuracy:.3f} Instance Accuracy {instance_accuracy:.3f}'
              .format(class_accuracy=test_class_acc, instance_accuracy=test_instance_acc))
    if return_accuracy_per_class:
        return test_class_acc, test_instance_acc, test_loss, class_acc
    return test_class_acc, test_instance_acc, test_loss
