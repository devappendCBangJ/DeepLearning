import torch.nn as nn

# torch.nn.Conv2d(
#     in_channels, 
#     out_channels, 
#     kernel_size, 
#     stride=1, 
#     padding=0, 
#     dilation=1, 
#     groups=1, 
#     bias=True, 
#     padding_mode='zeros'
# )

# Define model
class ConvNet(nn.Module):
    def __init__(self, drop_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 3, 2, 1)
        self.conv2 = nn.Conv2d(96, 192, 3, 1, 1)
        self.conv3 = nn.Conv2d(192, 384, 3, 2, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(384, 10)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = x.mean([-1, -2])
        x = self.drop(x)
        x = self.fc(x)
        return x