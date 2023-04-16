import os
import glob
import random
import torch

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image


def read_image(image):
    # image 불러오기 -> gray 변환
    image = Image.open(image)
    image = image.convert('L')
    return image


# 1) idx -> [Dataset] -> idx에 해당하는 data
# 2) dataset 묶음 -> [DataLoader] -> batch
# 3) batch로 내보내기 직전 후처리
    # - support_images : [N, Ks, 1, 32, 32]
    # - query_images : [N, Kq, 1, 32, 32]
    # - label : [N, Kq] -> label의 값을 형식에 맞게 바꿔야함
def omniglot_prototype_collate_fn(batched_data):
    support_images, query_images, query_labels = [], [], []
    # Fill this
    for i, (support_image, query_image, query_label) in enumerate(batched_data): # solution 방식!!! : query_label의 출력을 batch_size에 따라 (0)~(batch_size-1)이 되도록 변경
        support_images.append(support_image)
        query_images.append(query_image)
        query_labels.append(i * torch.ones_like(query_label, dtype=torch.long))
    support_images = torch.stack(support_images, dim=0) # 5차원 텐서 [N = batch_size, Ks, channel = 1, height = 32, width = 32]
    query_images = torch.stack(query_images, dim=0) # 5차원 텐서 [N = batch_size, Kq, channel = 1, height = 32, width = 32]
    query_labels = torch.stack(query_labels, dim=0) # 2차원 텐서 [N = batch_size, Kq]
    return support_images, query_images, query_labels


# 데이터셋 전처리
class OmniglotBaseline(Dataset):
    # class 시작 : image 불러오기 + 전처리 + 저장
    def __init__(self, root, K_s=5, K_q=1, training=False, transform=None):
        super().__init__()
        self.data = []
        # meta-test/alphabet에서 각 character에 해당되는 images 불러오기 + 전처리 + 저장
            # - data = [char1, char2, char3, ...] = alphabet에 존재하는 char list
            # - char1 = [image1, image2, ... , image10] = char에 존재하는 image list
        for character in os.listdir(root):
            images = glob.glob(f'{root}/{character}/*')
            images = map(read_image, images)
            images = map(to_tensor, images)
            images = torch.stack(list(images), dim=0)
            self.data.append(images)

        # 변수 저장
        self.transform = transform
        self.training = training
        self.K_s = K_s
        self.K_q = K_q
        # image index 저장
        if training:
            self.image_idx = [i for i in range(K_s)]
        else:
            self.image_idx = [i for i in range(K_s, K_s + K_q)]
    
    # class 자체 출력값 : char idx에 대응되는 image 반환 + label 반환
    def __getitem__(self, idx):
        # 원하는 char idx의 image 변환 + 반환
        K = len(self.image_idx)
        images = self.data[idx][self.image_idx, ...]
        if self.transform is not None:
            for k in range(K):
                images[k, ...] = self.transform(images[k, ...])
                
        # 원하는 image의 label 반환
        labels = idx * torch.ones((K,), dtype=torch.long) # ???
        return images, labels
    
    # class 개수 출력값 : image 개수
    def __len__(self):
        return len(self.data)


# 데이터셋 전처리
# 1) idx -> [Dataset] -> idx에 해당하는 data
# 2) dataset 묶음 -> [DataLoader] -> batch ->
    # (1) DataLoader 내장 sampler, batch_sampler : [B, C, H, W] 4차원 입력 사용해야함
    # (2) N way K shot : [B, N, K, C, H, W] 6차원 입력을 사용해야함
        # -> DataLoader 내장 sampler 사용 불가능
        # -> 데이터 형태 바꿔줘야함. 여기서는 trick을 통해 구현
class OmniglotPrototype(Dataset):
    # class 시작 : image 불러오기 + 전처리 + 저장
    def __init__(self, root, K_s=5, K_q=1, training=False, transform=None):
        super().__init__()
        self.data = []
        # meta-test/alphabet에서 각 character에 해당되는 image 불러오기 + 전처리 + 저장
            # - data = [char1, char2, char3, ...] = alphabet에 존재하는 char list
            # - char1 = [image1, image2, ... , image10] = char에 존재하는 image list
        for character in os.listdir(root):
            images = glob.glob(f'{root}/{character}/*')
            images = map(read_image, images)
            images = map(to_tensor, images)
            images = torch.stack(list(images), dim=0)
            self.data.append(images)

        # 변수 저장
        self.K_s = K_s
        self.K_q = K_q
        self.transform = transform
        self.image_idx = [i for i in range(20)]
        self.training = training

    # class 자체 출력값 : char idx에 대응되는 image 반환 + label 반환
    def __getitem__(self, idx):
        # 원하는 char idx의 images 변환 + 반환
        K = self.K_s + self.K_q
        if self.training: # train : 각 char 당 20개 image가 존재하는데, 이중에서 10개 image 랜덤 사용
            image_idx = random.sample(self.image_idx, k=K)
            images = self.data[idx][image_idx, ...]
        else: # test : 각 char 당 10개 image가 존재하는데, 모든 image 사용
            images = self.data[idx][:K, ...]

        if self.transform is not None:
            for k in range(K):
                images[k, ...] = self.transform(images[k, ...])

        # 원하는 char idx의 support image, query image, query label 반환 (각각 4차원 텐서 support_images = [Ks, channel = 1, height = 32, width = 32], query_images = [Kq, channel = 1, height = 32, width = 32])
        support_images, query_images = images.split([self.K_s, self.K_q], dim=0)
        query_labels = idx * torch.ones((self.K_q,), dtype=torch.long) # 사실 idx는 아무 의미 없는 값. 단순 데이터 구분용으로 사용
        return support_images, query_images, query_labels

    # class 개수 출력값 : image 개수
    def __len__(self):
        return len(self.data)