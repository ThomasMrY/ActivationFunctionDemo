from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch as th
import torch.optim as optim
import torchvision
import torch
from torchvision import datasets, transforms
import cv2
import numpy as np
import math
import NN_models.ops as ops
import os
import argparse
from NN_models.models import *
from NN_models.utils import progress_bar
from torch.nn.modules.module import Module
import NN_models.ops as ops


def train_test(training, file_name):
    class SELF_DEFINE(Module):

        def __init__(self, inplace=False):
            super(SELF_DEFINE, self).__init__()
            self.inplace = inplace

        def forward(self, input):
            return ops.self_define_torch(input)
    class SELF_DEFINE_APX(Module):
        def __init__(self, inplace=False):
            super(SELF_DEFINE_APX, self).__init__()
            self.inplace = inplace

        def forward(self, input):
            return ops.self_define_torch_apx(input)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=os.path.join('NN_models', 'CIFAR'), train=True, download=False,
                                            transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=os.path.join('NN_models', 'CIFAR'), train=False, download=False,
                                           transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('==> ...Building model...')
    net = VGG('VGG16')
    net.features[2] = SELF_DEFINE()
    net.features[5] = SELF_DEFINE()
    net.features[9] = SELF_DEFINE()
    net.features[12] = SELF_DEFINE()
    net.features[16] = SELF_DEFINE()
    net.features[19] = SELF_DEFINE()
    net.features[22] = SELF_DEFINE()
    net.features[26] = SELF_DEFINE()
    net.features[29] = SELF_DEFINE()
    net.features[32] = SELF_DEFINE()
    net.features[36] = SELF_DEFINE()
    net.features[39] = SELF_DEFINE()
    net.features[42] = SELF_DEFINE()
    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def test(epoch):
        global best_acc
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net.forward(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d)'
                             % (100. * correct / total, correct, total))
        return str(100. * correct / total)

    if (training == True):
        net.load_state_dict(torch.load(os.path.join('NN_models', 'CIFAR_data', 'CIFAR_tanh.pth')))
        for epoch in range(start_epoch, start_epoch + 200):
            train(epoch)
            torch.save(net.state_dict(), os.path.join('NN_models', 'CIFAR_data', 'CIFAR_tanh.pth'))
    else:
        net.load_state_dict(torch.load(os.path.join('NN_models', 'CIFAR_data', 'CIFAR_tanh.pth')))
        acc1 = test(0)
        net.features[2] = SELF_DEFINE_APX()
        net.features[5] = SELF_DEFINE_APX()
        net.features[9] = SELF_DEFINE_APX()
        net.features[12] = SELF_DEFINE_APX()
        net.features[16] = SELF_DEFINE_APX()
        net.features[19] = SELF_DEFINE_APX()
        net.features[22] = SELF_DEFINE_APX()
        net.features[26] = SELF_DEFINE_APX()
        net.features[29] = SELF_DEFINE_APX()
        net.features[32] = SELF_DEFINE_APX()
        net.features[36] = SELF_DEFINE_APX()
        net.features[39] = SELF_DEFINE_APX()
        net.features[42] = SELF_DEFINE_APX()
        acc2 = test(0)
        with open(os.path.join('NN_models', 'Acc', 'CIFAR_acc.txt'), 'a') as f:
            f.write(file_name + ' ')
            f.write(acc1 + ' ')
            f.write(acc2 + ' \r\n')


