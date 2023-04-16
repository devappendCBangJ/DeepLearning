import torch
import argparse

import easydict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchmetrics import MeanSquaredError

from src.autoencoders import Autoencoder
from src.engines import train, evaluate
from src.utils import load_checkpoint, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default="autoencoder")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--root", type=str, default="data")
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()

# Jupyter 환경
args = easydict.EasyDict({
        "title" : "autoencoder",
        "device" : "cpu",
        "root" : "data",
        "batch_size" : 100,
        "num_workers" : 2,
        "epochs" : 100,
        "lr" : 0.001,
        "logs": "logs",
        "checkpoints": "checkpoints",
        "resume": False
    })

def main(args):
    # Build dataset
    train_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(0.5, 0.25),
        T.Lambda(torch.flatten)
    ])
    train_data = FashionMNIST(root=args.root, train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(0.5, 0.25),
        T.Lambda(torch.flatten)
    ])
    val_data = FashionMNIST(root=args.root, train=False, download=True, transform=val_transform)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # Build model
    model = Autoencoder()
    model = model.to(args.device)

    # Build optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Build scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader))

    # Build loss function
    loss_fn = nn.MSELoss()

    # Build metric function
    metric_fn = MeanSquaredError()

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
        train_logger.add_scalar('MSE', train_summary['metric'], epoch + 1)
        val_logger.add_scalar('Loss', val_summary['loss'], epoch + 1)
        val_logger.add_scalar('MSE', val_summary['metric'], epoch + 1)

        # save model
        save_checkpoint(args.checkpoints, args.title, model, optimizer, epoch + 1)

    # Fill this
    # 1. you should build data processing pipe line to train an autoencoder
    #    * transforms for data processing pipe line 
    #    - ToTensor (PIL image -> torch tensor)
    #    - Reshape (28 x 28 -> 784)
    #    - Normalize Tensors (mean=0.5, std=0.25)
    #    - Add Gaussian noise (mean=0.0, std=0.25)
    # 2. you should build the autoencoder in src.autoencoders.py
    # 3. you should build performenace metric (MSE), loss function (MSE), 
    #    optimizer (Adam), and learning rate scheduler (CosineSchedule)
    # 4. you should build training and evaluation loop
    # 5. you should train the autoencoder
    # 6. you should save the learning parameters of autoencoder

if __name__=="__main__":
    main(args)