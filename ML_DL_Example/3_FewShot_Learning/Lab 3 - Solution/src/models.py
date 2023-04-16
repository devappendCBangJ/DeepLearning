import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_dim, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, dim, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            Block(1, 32, 3, 2, 1),
            Block(32, 64, 3, 2, 1),
            Block(64, 128, 3, 2, 1),
            Block(128, 256, 3, 1, 1),
        )
        
    def forward(self, x):
        x = self.layers(x)
        x = x.mean([-1, -2])
        return x


class BaselineNet(nn.Module):
    def __init__(self, num_classes=1432, pretrain=False):
        super().__init__()
        self.features = EmbeddingNet()
        self.head = nn.Linear(256, num_classes)
        self.pretrain = pretrain
    
    def forward(self, x):
        if not self.pretrain:
            N, K, C, H, W = x.shape
            x = x.reshape((N * K, C, H, W))
        x = self.features(x)
        x = self.head(x)
        if not self.pretrain:
            x = x.reshape((N, K, -1))
        return x


class PrototypeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = EmbeddingNet()
    
    def forward(self, x_s, x_q):
        # Fill this
        z_s = self.extract_features(x_s)
        z_q = self.extract_features(x_q)
        c_s = self.compute_prototype(z_s)
        y_q = -self.compute_distance(z_q, c_s)
        return y_q

    def extract_features(self, x):
        N, K, C, H, W = x.shape
        z = x.reshape((N * K, C, H, W))
        z = self.features(z)
        z = z.reshape((N, K, -1))
        return z

    def compute_prototype(self, z_s):
        # Fill this
        c_s = z_s.mean(dim=1)
        return c_s

    def compute_distance(self, z_q, c_s):
        # Fill this
        N, K, D = z_q.shape
        z_q = z_q.reshape((N, K, 1, D))
        c_s = c_s.reshape((1, 1, N, D))
        d = ((z_q - c_s) ** 2).sum(dim=-1)
        # The implementation using torch.cdist or torch.nn.functional.pairwise_distance is also possible 
        # z_q = z_q.reshape((N * K, D))
        # d = torch.cdist(z_q, c_s, p=2)
        # d = d.reshape(N, K, D)
        # These functions will return actural Euclidean distances instead of their squared ones
        return d
    