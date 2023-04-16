import torch
import argparse
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from src.kmeans import kmeans
from src.metrics import clustering_accuracy
from src.autoencoders import Autoencoder


parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default="autoencoder")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--root", type=str, default="data")
parser.add_argument("--checkpoints", type=str, default='checkpoints')
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--num_clusters", type=int, default=10)
parser.add_argument("--num_iterations", type=int, default=20)
args = parser.parse_args()

def extract_feature(loader, model, device):
    model.eval()
    features = torch.empty((0, 100), dtype=torch.float32)
    labels = torch.empty((0, ), dtype=torch.long)
    for image, label in loader:
        image = image.to(device)
        with torch.no_grad():
            feature, _ = model(image)
        feature = feature.to('cpu')
        features = torch.cat([features, feature], dim=0)
        labels = torch.cat([labels, label], dim=0)
    return features, labels

def main(args):
    transform = T.Compose([
        T.ToTensor(),
        # T.Normalize((0.5), (0.25)),
    ])
    dataset = FashionMNIST(args.root, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Fill this
    # 1. you should build the autoencoder
    # 2. you should load pretrained weights to the autoencoder 
    # 3. you should create 60,000 x 100 latent representations of examples (features) and 60,000 targets

    model = Autoencoder()
    model = model.to(args.device)
    
    checkpoint_path = f'{args.checkpoints}/{args.title}.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    features, targets = extract_feature(loader, model, args.device)
    
    _, predictions = kmeans(features, args.num_clusters, args.num_iterations)
    print(predictions, targets)
    accuracy = clustering_accuracy(predictions, targets, args.num_clusters)
    print(f'K: {args.num_clusters}, Acc.: {accuracy:.4f}')



if __name__=="__main__":
    main(args)