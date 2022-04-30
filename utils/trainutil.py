import glob
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from augmentations.augmentations import Compose, RandomRotate, Scale
import torchvision.transforms as transforms


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    epoch += 1
    if epoch <= args.warmup:
        lr = args.learning_rate * epoch / args.warmup
    elif epoch > args.steps[1]:
        lr = args.learning_rate * 0.01
    elif epoch > args.steps[0]:
        lr = args.learning_rate * 0.1
    else:
        lr = args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    return acc


def train(epoch, data_loader, model, optimizer, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    # for train_features, train_labels in next(iter(data_loader))
    for idx, (data, target) in enumerate(data_loader):
        start = time.time()

        if torch.cuda.is_available():
            # tensor_dtype = torch.cuda.FloatTensor
            # print('chanding dtype of input & target tensors')
            data = data.cuda()
            target = target.cuda()

        print(data.shape, criterion)
        out = model.forward(data)
        loss = criterion(out, target)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss, out.shape[0])
        # batch_acc = accuracy(out, target)
        # acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 1 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
                  .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses, top1=acc))


def validate(epoch, val_loader, model, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    num_class = 10
    # cm = torch.zeros(num_class, num_class)
    # evaluation loop
    model.eval()
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target)

        # batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        # for t, p in zip(target.view(-1), preds.view(-1)):
        #     cm[t.long(), p.long()] += 1

        losses.update(loss, out.shape[0])
        # acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t')
                  .format(epoch, idx, len(val_loader), iter_time=iter_time, loss=losses, top1=acc))
    # cm = cm / cm.sum(1)
    # per_cls_acc = cm.diag().detach().numpy().tolist()
    # for i, acc_i in enumerate(per_cls_acc):
    #     print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    # print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
    return losses.avg