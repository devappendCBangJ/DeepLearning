import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchmetrics.aggregation import MeanMetric
from src.autoencoders import Autoencoder


parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default="autoencoder_")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--checkpoints", type=str, default='checkpoints')
parser.add_argument("--root", type=str, default="data")
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()


def random_noise(image, std=0.25):
    image = image + torch.randn_like(image) * std
    image = torch.clip(image, 0.0, 1.0)
    return image


def train(loader, model, optimizer, scheduler, loss_fn, device):
    model.train()
    loss_mean = MeanMetric()
    for x_clean, _ in loader:
        # Make noisy image (input)
        x_noise = random_noise(x_clean)
        x_noise = x_noise.to(device)
        x_clean = x_clean.to(device)

        # Forward propagation
        _, x_recon = model(x_noise)
        loss = loss_fn(x_recon, x_clean)
        # Backward propagation
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()
        scheduler.step()
        # Update statistics 
        loss_mean.update(loss.to('cpu'))

    summary = {'loss': loss_mean.compute()}
    return summary


def save_checkpoint(checkpoint_dir, title, model, optimizer, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_dict = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    checkpoint_path = f'{checkpoint_dir}/{title}.pth'
    torch.save(state_dict, checkpoint_path)


def main(args):
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

    # Build dataset
    # We implement "Add Gaussian noise" at training loop. Please see random_noise and train functions
    # We implement "Reshape" at our model. Please see AutoEncoder in src/autoencoders.py 
    transform = T.Compose([
        T.ToTensor(),
        # T.Normalize((0.5), (0.25)),
    ])
    dataset = FashionMNIST(args.root, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # Build model
    model = Autoencoder()
    model = model.to(args.device)
    # In this code, we train our model on cpu, but if we use gpu (Google Colab), the code will run much faster

     # Build optimizer 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Build scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(loader))

    # Build loss function
    loss_fn = nn.MSELoss()

    # Main loop
    for epoch in range(args.epochs):
        # train one epoch
        train_summary = train(loader, model, optimizer, scheduler, loss_fn, args.device)
        print('MSE', train_summary['loss'], epoch + 1)
        
        # save checkpoint
        save_checkpoint(args.checkpoints, args.title, model, optimizer, epoch + 1)


if __name__=="__main__":
    main(args)