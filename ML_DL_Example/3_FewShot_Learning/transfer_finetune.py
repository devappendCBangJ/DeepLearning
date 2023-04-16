# Colab 환경
import argparse

# Jupyter 환경
import easydict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import numpy as np

from torch.utils.data import DataLoader
from torchmetrics.functional.classification import accuracy

from src.datasets import OmniglotBaseline
from src.models import BaselineNet
from src.engines import train_baseline, evaluate_baseline
from src.utils import save_checkpoint

# # Colab 환경
# parser = argparse.ArgumentParser()
# parser.add_argument("--title", type=str, default="transfer")
# parser.add_argument("--device", type=str, default="cuda")
# parser.add_argument("--root", type=str, default="data/omniglot/meta-test")
# parser.add_argument("--num_workers", type=int, default=2)
# parser.add_argument("--alphabets", type=str, nargs=5, default=["Atlantean", "Japanese_(hiragana)", "Japanese_(katakana)", "Korean", "ULOG"])
# parser.add_argument("--num_characters", type=int, nargs=5, default=[26, 52, 47, 40, 26])
# parser.add_argument("--num_supports", type=int, default=5)
# parser.add_argument("--num_queries", type=int, default=5)
# parser.add_argument("--batch_size", type=int, default=16)
# parser.add_argument("--epochs", type=int, default=200)
# parser.add_argument("--lr", type=float, default=0.001)
# parser.add_argument("--checkpoints", type=str, default='checkpoints')
# parser.add_argument("--pretrain", type=bool, default=False)
# args = parser.parse_args()

# Jupyter 환경
args = easydict.EasyDict({
    "title" : "transfer",
    "device" : "cuda",
    "root" : "data/omniglot/meta-test",
    "num_workers" : 2,
    "alphabets" : ["Atlantean", "Japanese_(hiragana)", "Japanese_(katakana)", "Korean", "ULOG"],
    "num_characters" : [26, 52, 47, 40, 26],
    "num_supports" : 5,
    "num_queries" : 5,
    "batch_size" : 16,
    "epochs" : 200,
    "lr" : 0.001,
    "checkpoints" : 'checkpoints',
    "pretrain" : False
})

def main(args):
    accuracies = []

    for alphabet, num_classes in zip(args.alphabets, args.num_characters):
        # Build dataset
        root = f'{args.root}/{alphabet}'
        train_data = OmniglotBaseline(root, args.num_supports, args.num_queries, training=True, transform=T.RandomCrop((32, 32), padding=4))
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        val_data = OmniglotBaseline(root, args.num_supports, args.num_queries, training=False)
        val_loader = DataLoader(val_data, batch_size=num_classes, num_workers=args.num_workers)

        # Build model
        model = BaselineNet(num_classes, args.pretrain)

        # fill this
            # - pytorch 모듈 파라미터 불러오기 : pytorch 설명 ppt, lab1, lab2 참조
            # - train된 network의 마지막 layer에 pretrain 파라미터를 가져와야함
            # - BaselineNet의 head : class의 개수가 다르므로 불러오면 안됨. [pretrain class 개수 - 1432개], [test class 개수 - 20way 5shot이므로 20개]
            # - BaselineNet의 features : 불러오기
        checkpoint_path = f'{args.checkpoints}/{args.title}_embedding.pth'
        state_dict = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)

        model = model.to(args.device)

        # Build optimizer 
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader))
        loss_fn = nn.CrossEntropyLoss()
        metric_fn = accuracy

        # Main loop
        for epoch in range(args.epochs):
            train_summary = train_baseline(train_loader, model, optimizer, scheduler, loss_fn, metric_fn, args.device)
            val_summary = evaluate_baseline(val_loader, model, loss_fn, metric_fn, args.device)
            print(f'Epoch: {epoch + 1}, Train Accuracy: {train_summary["metric"]:.4f}, Val Accuracy: {val_summary["metric"]:.4f}')
            save_checkpoint(args.checkpoints, f'{args.title}-{alphabet}', model, optimizer, epoch + 1)
        accuracies.append(val_summary["metric"])
    
    # Print performance
    for i, alphabet in enumerate(args.alphabets):
        print(f'{alphabet}: {accuracies[i]:.4f}')
    mean_accuracy = np.mean(accuracies)
    mean_std = np.std(accuracies)
    print(f'mean: {mean_accuracy:.4f}, std: {mean_std:.4f}')


if __name__=="__main__":
    main(args)