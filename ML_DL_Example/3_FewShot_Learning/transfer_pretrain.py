import argparse
import easydict

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import accuracy

from src.engines import pretrain_baseline
from src.models import BaselineNet
from src.utils import save_checkpoint, save_pretrained_embeddingnet

# Colab 환경
# parser = argparse.ArgumentParser()
# parser.add_argument("--title", type=str, default="transfer")
# parser.add_argument("--device", type=str, default="cuda")
# parser.add_argument("--root", type=str, default="data/omniglot/meta-train")
# parser.add_argument("--num_workers", type=int, default=2)
# parser.add_argument("--num_classes", type=int, default=1432)
# parser.add_argument("--batch_size", type=int, default=16)
# parser.add_argument("--epochs", type=int, default=100)
# parser.add_argument("--lr", type=float, default=0.001)
# parser.add_argument("--checkpoints", type=str, default='checkpoints')
# parser.add_argument("--pretrain", type=bool, default=True)
# args = parser.parse_args()

# Jupyter 환경
args = easydict.EasyDict({
    "title" : "transfer",
    "device" : "cuda",
    "root" : "data/omniglot/meta-train",
    "num_workers" : 2,
    "num_classes" : 1432,
    "batch_size" : 16,
    "epochs" : 100,
    "lr" : 0.001,
    "checkpoints" : 'checkpoints',
    "pretrain" : True
})


def main(args):
    # Build dataset : train만 진행
    train_transform = T.Compose([
        T.Grayscale(),
        T.RandomCrop((32, 32), padding=4),
        T.ToTensor(),
    ])
    train_data = ImageFolder(args.root, transform=train_transform)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # Build model
    model = BaselineNet(args.num_classes, pretrain=args.pretrain)

    model = model.to(args.device)

    # Build optimizer 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = accuracy

    # Main loop : train만 진행
    for epoch in range(args.epochs):
        train_summary = pretrain_baseline(train_loader, model, optimizer, scheduler, loss_fn, metric_fn, args.device)
        
        print(f'Epoch: {epoch + 1}, Train Accuracy: {train_summary["metric"]:.4f}')
        save_checkpoint(args.checkpoints, args.title, model, optimizer, epoch + 1)
    
    save_pretrained_embeddingnet(args.checkpoints, args.title, model.features) # model이 아닌, model.features를 인수로 사용 ♣


if __name__=="__main__":
    main(args)