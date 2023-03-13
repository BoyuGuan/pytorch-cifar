
'''Train CIFAR10 with PyTorch.'''
import os
import argparse
import time 
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision
import torch.multiprocessing as mp
import torchvision.transforms as transforms
# from  models import *
import torchvision.models

from models.cifar10 import *
from utils import reduce_mean



logger = logging.getLogger('myTrain')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--batch_size','--batch-size', default=5120, type=int)
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--ip', default='127.0.0.1', type=str)
parser.add_argument('--port', default='23456', type=str)



def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    # print('11111\n')
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank
    init_seeds(local_rank+1, False) 
    init_method = 'tcp://' + args.ip + ':' + args.port
    cudnn.benchmark = True
    # print(f'{args.nprocs},   {local_rank}\t\n')
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=args.nprocs,
                            rank=local_rank)
    model = CIFAR10Model()
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    batch_size = int(args.batch_size / nprocs) # 需要手动划分 batch_size 为 mini-batch_size

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, pin_memory = True,sampler=train_sampler)

    beginTime = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for step, (images, labels) in enumerate(train_loader):
            # 将对应进程的数据放到对应GPU上
            images = images.cuda(local_rank, non_blocking = True)
            labels = labels.cuda(local_rank, non_blocking = True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            torch.distributed.barrier() # 同步一波
            reduced_loss = reduce_mean(loss, args.nprocs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.local_rank == 0:
                print(
                    'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                        reduced_loss,
                        optimizer.param_groups[0]['lr'],
                        epoch=epoch+1,
                        trained_samples=step * args.batch_size + len(images),
                        total_samples=len(train_loader.dataset)
                    ))
        train_scheduler.step()
    endTime = time.time()
    if args.local_rank == 0:
        print(f'\n3 epochs train time cost is {endTime - beginTime}')

        

if __name__ == '__main__':
    os.makedirs('./log',exist_ok=True)
    fileHandler = logging.FileHandler('./log/multi.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)

    main()
