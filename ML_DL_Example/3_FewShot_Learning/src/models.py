import torch.nn as nn

class Block(nn.Module):
    # Block : Conv2d -> 정규화 -> ReLU
    def __init__(self, in_dim, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, dim, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(dim) # 인수가 왜 dim???
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

# EmbeddingNet : Block[1, 32] -> Block[32, 64] -> Block[64, 128] -> Block[128, 256]
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
        x = x.mean([-1, -2]) # 3. global image pooling 출력 = 2차원 텐서 : (batch, channel, height, width) -> (batch, channel = 256) ♣
        return x

# BaselineNet : EmbeddingNet -> num_classes
class BaselineNet(nn.Module): # module이기 때문에 파라미터 저장 가능 ♣
    def __init__(self, num_classes=1432, pretrain=False):
        super().__init__()
        self.features = EmbeddingNet()
        self.head = nn.Linear(256, num_classes)
        self.pretrain = pretrain
    
    def forward(self, x):
        if not self.pretrain: # transfer learning이 아닌, meta learning인 경우
            N, K, C, H, W = x.shape # 1. finetune image classification 출력 = 5차원 텐서(N way K shot 형태) : (N way, K shot, channel, height, width) ♣
            x = x.reshape((N * K, C, H, W)) # 2. Convolution layer 입력 = 4차원 텐서 : (batch, channel, height, width) ... N K가 다르면 서로 다른 이미지이기 때문에 batch로 사용 가능 ♣
        x = self.features(x)
        x = self.head(x)
        if not self.pretrain: # transfer learning이 아닌, meta learning인 경우
            x = x.reshape((N, K, -1)) # 4. finetune image classification 출력 = 3차원 텐서(N way K shot 형태) : (batch, num_classes) -> (N way, K shot, num_classes) ♣
        return x


class PrototypeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = EmbeddingNet()

    # training loop 반복할 때마다, support_images, query_images 받아들임. query_label은 나중에 loss 계산할 때 사용
    # embedding network 활용해서 feature로 만듦
    def forward(self, x_s, x_q):
        # Fill this
        # solution 풀이!!!
        z_s = self.extract_features(x_s)
        z_q = self.extract_features(x_q)
        c_s = self.compute_prototype(z_s)
        y_q = -self.compute_distance(z_q, c_s)
        return y_q
    
    def extract_features(self, x):
        # solution 풀이!!!
        N, K, C, H, W = x.shape
        z = x.reshape((N * K, C, H, W))
        z = self.features(z)
        z = z.reshape((N, K, -1))
        return z

    # support_example에 대한 feature를 이용해서 prototype 만들어냄
    def compute_prototype(self, z_s):
        # Fill this
        # solution 풀이!!!
        c_s = z_s.mean(dim=1)
        return c_s

    # prototype과 query_example에 대한 feature를 유클리디안 distance를 활용한 거리 계산 -> 음수값을 씌워서 return -> cross entrophy loss 정의 가능(softmax는 나중에 loss function 내에서 취함)
    def compute_distance(self, z_q, c_s):
        # Fill this
        # solution 풀이!!!
        N, K, D = z_q.shape
        z_q = z_q.reshape((N, K, 1, D))
        c_s = c_s.reshape((1, 1, N, D))
        d = ((z_q - c_s) ** 2).sum(dim=-1)
        return d

    # DataLoader 출력 : 5차원 텐서 [N, K, 1, 32, 32]
    # -> reshape 필수