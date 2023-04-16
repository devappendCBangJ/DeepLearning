import argparse

from torchvision.datasets import FashionMNIST
from src.kmeans import kmeans
from src.metrics import clustering_accuracy
from einops import rearrange

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default='data')
parser.add_argument("--num_clusters", type=int, default=10)
parser.add_argument("--num_iterations", type=int, default=20)
args = parser.parse_args()

def load_fashionmnist(root):
    dataset = FashionMNIST(root=root, train=True, download=True)
    examples, targets = dataset.data, dataset.targets
    examples = examples.float() / 255.0 
    # examples = rearrange(examples, 'n h w -> n 1 h w')
    # examples = normalize(examples, mean=(0.5,), std=(0.25,))
    examples = rearrange(examples, 'n h w -> n (h w)')
    return examples, targets


def main(args):
    examples, targets = load_fashionmnist(args.root)
    _, predictions = kmeans(examples, args.num_clusters, args.num_iterations)
    accuracy = clustering_accuracy(predictions, targets, args.num_clusters)
    print(f'K: {args.num_clusters}, Acc.: {accuracy:.4f}')


if __name__=="__main__":
    main(args)