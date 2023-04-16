import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchmetrics.aggregation import MeanMetric
from torchmetrics.functional.classification import accuracy

from src.models import ConvNet
from src.engines import train, evaluate
from src.utils import load_checkpoint, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default="regularization")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--root", type=str, default="data")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--label_smoothing", type=float, default=0.05)
parser.add_argument("--drop_rate", type=float, default=0.1)
parser.add_argument("--logs", type=str, default='logs')
parser.add_argument("--checkpoints", type=str, default='checkpoints')
parser.add_argument("--resume", type=bool, default=False)
args = parser.parse_args()

def main(args):
    # Build dataset
    train_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_data = CIFAR10(args.root, train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    val_data = CIFAR10(args.root, train=False, download=True, transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers)

    # Build model
    model = ConvNet(drop_rate=args.drop_rate)
    model = model.to(args.device)

    # Build optimizer 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Build scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader))

    # Build loss function
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Build metric function
    metric_fn = accuracy

    # Build logger
    train_logger = SummaryWriter(f'{args.logs}/train/{args.title}')
    val_logger = SummaryWriter(f'{args.logs}/val/{args.title}')

    # Load model
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.checkpoints, args.title, model, optimizer)
    
    # Main loop
    for epoch in range(start_epoch, args.epochs):
        # train one epoch
        train_summary = train(train_loader, model, optimizer, scheduler, loss_fn, metric_fn, args.device)
        
        # evaluate one epoch
        val_summary = evaluate(val_loader, model, loss_fn, metric_fn, args.device)

        # write log
        train_logger.add_scalar('Loss', train_summary['loss'], epoch + 1)
        train_logger.add_scalar('Accuracy', train_summary['metric'], epoch + 1)
        val_logger.add_scalar('Loss', val_summary['loss'], epoch + 1)
        val_logger.add_scalar('Accuracy', val_summary['metric'], epoch + 1)

        # save model
        save_checkpoint(args.checkpoints, args.title, model, optimizer, epoch + 1)

if __name__=="__main__":
    main(args)