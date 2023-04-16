import argparse

from torchvision.datasets import FashionMNIST
from torchvision.transforms.functional import normalize
from src.kmeans import kmeans
from src.metrics import clustering_accuracy
from einops import rearrange

from src.all_print import all_print

# 변수 선언
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default='data')
parser.add_argument("--num_clusters", type=int, default=10)
parser.add_argument("--num_iterations", type=int, default=20)
args = parser.parse_args()

# 데이터 불러오기 + 전처리
def load_fashionmnist(root):
    dataset = FashionMNIST(root=root, train=True, download=True) # [dataset] type : torchvision.datasets.mnist.FashionMNIST / len : 60000
    # print("dataset_type : ", type(dataset))
    # print("dataset_len : ", len(dataset))
    # print("dataset : ", dataset)
    examples, targets = dataset.data, dataset.targets # [examples] type : torch.Tensor / len : 60000 / shape : torch.Size([60000, 28, 28]) /// # [targets] type : torch.Tensor / len : 60000 / shape : torch.Size([60000])
    # print("examples_type : ", type(examples))
    # print("examples_len : ", len(examples))
    # print("examples_shape : ", examples.shape)
    # print("examples : ", examples)
    # all_print(examples, locals())

    # print("targets_type : ", type(targets))
    # print("targets_len : ", len(targets))
    # print("targets_shape : ", targets.shape)
    # print("targets : ", targets)
    examples = examples.float() # [examples] type : torch.Tensor, len : 60000, shape : torch.Size([60000, 28, 28])
    # print("examples_type : ", type(examples))
    # print("examples_len : ", len(examples))
    # print("examples_shape : ", examples.shape)
    # print("examples : ", examples)
    examples = rearrange(examples, 'n h w -> n 1 h w') # [examples] type : torch.Tensor, len : 60000, shape : torch.Size([60000, 1, 28, 28])
    # print("examples_type : ", type(examples))
    # print("examples_len : ", len(examples))
    # print("examples_shape : ", examples.shape)
    # print("examples : ", examples)
    examples = normalize(examples, mean=(0.5,), std=(0.25,)) # [examples] type : torch.Tensor, len : 60000, shape : torch.Size([60000, 1, 28, 28])
    # print("examples_type : ", type(examples))
    # print("examples_len : ", len(examples))
    # print("examples_shape : ", examples.shape)
    # print("examples : ", examples)
    examples = rearrange(examples, 'n 1 h w -> n (1 h w)') # [examples] type : torch.Tensor, len : 60000, shape : torch.Size([60000, 784])
    # print("examples_type : ", type(examples))
    # print("examples_len : ", len(examples))
    # print("examples_shape : ", examples.shape)
    # print("examples : ", examples)
    return examples, targets

def main(args):
    # 데이터 불러오기 + 전처리
    examples, targets = load_fashionmnist(args.root) # [examples] type : torch.Tensor / len : 60000 / shape : torch.Size([60000, 784]) /// # [targets] type : torch.Tensor / len : 60000 / shape : torch.Size([60000])
    # print("examples_type : ", type(examples))
    # print("examples_len : ", len(examples))
    # print("examples_shape : ", examples.shape)
    # print("examples : ", examples)

    # print("targets_type : ", type(targets))
    # print("targets_len : ", len(targets))
    # print("targets_shape : ", targets.shape)
    # print("targets : ", targets)

    # K-means 알고리즘 실행
    _, predictions = kmeans(examples, args.num_clusters, args.num_iterations)
    accuracy = clustering_accuracy(predictions, targets, args.num_clusters)
    print(f'K: {args.num_clusters}, Acc.: {accuracy:.4f}')

if __name__=="__main__":
    for i in range(5):
        main(args)