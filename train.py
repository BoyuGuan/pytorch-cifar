
'''Train CIFAR10 with PyTorch.'''
import os
import argparse
import time 
import logging

import torch
import torch.nn as nn
import torch.optim as optim

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



import torchvision
import torchvision.transforms as transforms
# from  models import *
import torchvision.models


from utils import progress_bar
from models.cifar10 import * 

logger = logging.getLogger('cifarTrain')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
args = parser.parse_args()
best_acc = 0  # best test accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepareData(batchSize = 1024):
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

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainDataLoader = torch.utils.data.DataLoader(
        trainset, batch_size=batchSize, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testDataLoader = torch.utils.data.DataLoader(
        testset, batch_size=batchSize, shuffle=False, num_workers=8)
    return trainDataLoader, testDataLoader


# Training
def train(net, epochs, trainDataLoader, testDataLoader):
    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    beginTime = time.time()
    for epoch in range(epochs):
        print('\nEpoch: %d' % (epoch+1))
        
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainDataLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar(batch_idx, len(trainDataLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        scheduler.step()
    endTime = time.time()
    logger.info(f'3 epochs time is {endTime - beginTime}')
    test(net, testDataLoader, criterion)



def test(net, testDataLoader, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testDataLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testDataLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save Model
    acc = 100.*correct/total
    if acc > best_acc:
        torch.save(net, './trainedModel/mobilenet.pth')
        best_acc = acc



if __name__ == '__main__':
    os.makedirs('./log',exist_ok=True)
    fileHandler = logging.FileHandler('./log/singleCard_CNN.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)

    trainDataLoader, testDataLoader = prepareData()
    net = torchvision.models.resnet50()
    train(net, 3, trainDataLoader, testDataLoader)
    logger.info(f'max GPU memory allocated {torch.cuda.max_memory_allocated()/1024/1024/1024} GB')