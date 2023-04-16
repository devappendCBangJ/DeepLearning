import torch.nn as nn

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec

# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 100)
        self.act = nn.ReLU()
        
    def forward(self, x):
        print("x.shape : ", x.shape)
        # x = x.view(x.size(0), -1)
        # print("x.shape : ", x.shape)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        print("x.shape : ", x.shape)
        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 250)
        self.fc2 = nn.Linear(250, 500)
        self.fc3 = nn.Linear(500, 784)
        self.act = nn.ReLU()
        
    def forward(self, x):
        print("x.shape : ", x.shape)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        print("x.shape : ", x.shape)
        return x