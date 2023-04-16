import torch
import argparse

from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from src.kmeans import kmeans
from src.metrics import clustering_accuracy


parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default="autoencoder")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--root", type=str, default="data")
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--num_clusters", type=int, default=10)
parser.add_argument("--num_iterations", type=int, default=20)
args = parser.parse_args()

def main(args):
    dataset = FashionMNIST(root=args.root, train=True, download=True)
    loader = DataLoader(dataset, args.batch_size, num_workers=args.num_workers)

    # Fill this
    # 1. you should build the autoencoder
    # 2. you should load pretrained weights to the autoencoder 
    # 3. you should create 60,000 x 100 latent representations of examples (features) and 60,000 targets

    _, predictions = kmeans(features, args.num_clusters, args.num_iterations)
    accuracy = clustering_accuracy(predictions, targets, args.num_clusters)
    print(f'K: {args.num_clusters}, Acc.: {accuracy:.4f}')



if __name__=="__main__":
    main(args)