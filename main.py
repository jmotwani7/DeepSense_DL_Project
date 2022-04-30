import os

import numpy as np
# import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.utils.data as Data
import argparse
import yaml
import copy
import torchvision
import torchvision.transforms.functional as tf
import pandas as pd
import torch.nn as nn
# from models import resnet32,UNet
from tqdm import tqdm
import time
import glob
from augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale
from model import baseModel

np.random.seed(100)


class NyuGenerator(torch.utils.data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=(480, 640), augmentations=None, img_norm=True):
        self.root = root
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_size = img_size
        self.split = split
        self.files = {}
        for split in ["train", "test"]:
            file_list = glob.glob(root + "/" + split + "/" + split + "_images/**")
            # print(file_list)
            self.files[split] = file_list

    def __len__(self):
        # return int(np.ceil(len(self.data) / self.batch_size))
        return len(self.files[self.split])

    def __getitem__(self, id):
        img_path = self.files[self.split][id]
        img_name = os.path.split(self.files[self.split][0])[-1][:-4]
        label_path = root + "/" + self.split + "/" + self.split + "_labels/" + img_name + ".png"

        img = np.asarray(Image.open(img_path))

        label = np.asarray(Image.open(label_path))

        if self.is_transform:
            img, label = self.transform(img, label)

        if self.augmentations is not None:
            img, label = self.augmentations(img, label)

        return img, label

    def transform(self, img, label):
        img = Image.fromarray(img).resize((480, 640), Image.ANTIALIAS)
        label = Image.fromarray(label).resize((480, 640), Image.ANTIALIAS)

        return np.asarray(img), np.asarray(label)


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


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    return acc


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


def train(epoch, data_loader, model, optimizer, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    model.train()
    for idx, (data, target) in enumerate(data_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        print(data.shape, criterion)
        out = model(data)
        loss = criterion.forward(out, target)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(out, target)

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
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
    cm = torch.zeros(num_class, num_class)
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

        batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t')
                  .format(epoch, idx, len(val_loader), iter_time=iter_time, loss=losses, top1=acc))
    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
    return acc.avg, cm

parser = argparse.ArgumentParser(description='DeepSense_DL_Project')
parser.add_argument('--config', default='./config_resnet.yaml')

global args
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f)

for key in config:
    for k, v in config[key].items():
        setattr(args, k, v)
model = baseModel.ResNet50Base()

optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                            momentum=args.momentum,
                            weight_decay=args.reg)
best = 0.0
best_cm = None
best_model = None
for epoch in range(args.epochs):
    adjust_learning_rate(optimizer, epoch, args)

    # train loop
    train(epoch, train_loader, model, optimizer, criterion)

    # validation loop
    acc, cm = validate(epoch, test_loader, model, criterion)

    if acc > best:
        best = acc
        best_cm = cm
        best_model = copy.deepcopy(model)

print('Best Prec @1 Acccuracy: {:.4f}'.format(best))
per_cls_acc = best_cm.diag().detach().numpy().tolist()
for i, acc_i in enumerate(per_cls_acc):
    print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

if args.save_best:
    torch.save(best_model.state_dict(), './checkpoints/' + args.model.lower() + '.pth')
