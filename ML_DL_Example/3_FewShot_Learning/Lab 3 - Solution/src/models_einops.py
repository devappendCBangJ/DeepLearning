import torch.nn as nn

from einops import rearrange, reduce
from einops.layers.torch import Reduce

class Block(nn.Module):
    def __init__(self, in_dim, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, dim, kernel_size, stride, padding),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x


class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            Block(1, 32, 3, 2, 1),
            Block(32, 64, 3, 2, 1),
            Block(64, 128, 3, 2, 1),
            Block(128, 256, 3, 1, 1),
            Reduce('n c h w -> n c', 'mean'),
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x


class BaselineNet(nn.Module):
    def __init__(self, num_classes=1432, pretrain=False):
        super().__init__()
        self.features = EmbeddingNet()
        self.head = nn.Linear(256, num_classes)
        self.pretrain = pretrain
    
    def forward(self, x):
        if not self.pretrain:
            z = rearrange(x, 'n k c h w -> (n k) c h w')
        z = self.features(z)
        z = self.head(z)
        if not self.pretrain:
            z = rearrange(z, '(n k) c -> n k c', n=x.shape[0])
        return z


class PrototypeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = EmbeddingNet()
    
    def forward(self, x_s, x_q):
        z_s = self.extract_features(x_s)
        z_q = self.extract_features(x_q)
        c_s = self.compute_prototype(z_s)
        y_q = -self.compute_distance(z_q, c_s)
        return y_q

    def extract_features(self, x):
        z = rearrange(x, 'n k c h w -> (n k) c h w')
        z = self.features(z)
        z = rearrange(z, '(n k) c -> n k c', n=x.shape[0])
        return z

    def compute_prototype(self, z_s):
        c_s = reduce(z_s, 'n k c -> n c', 'mean')
        return c_s

    def compute_distance(self, z_q, c_s):
        z_q = rearrange(z_q, 'n k c -> n k 1 c')
        c_s = rearrange(c_s, 'n c -> 1 1 n c')
        d = ((z_q - c_s) ** 2).sum(dim=-1)
        return d
    