import torch
import torch.nn as nn

from einops import rearrange

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        x = rearrange(x, 'n c h w -> n (c h w)')
        z = self.encoder(x)
        x_rec = self.decoder(z)
        x_rec = rearrange(x_rec, 'n (c h w) -> n c h w', c=1, h=28, w=28)
        return z, x_rec

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 100)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 250)
        self.fc2 = nn.Linear(250, 500)
        self.fc3 = nn.Linear(500, 784)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x