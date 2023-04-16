import torch
import random
import numpy as np
from einops import rearrange, reduce

from .all_print import all_print

def kmeans(examples, K, T):
    """ kmeans clustering
    Args:
        examples: (torch.Tensor) N x D tensor
        K: (int) The number of clusters
        T: (int) The number of iterations
    Returns:
        clusters: (torch.Tensor) K x D tensor
        predictions: (torch.Tensor) N tensor
    """
    
    # # K-means cluster 초기화
    # clusters = init_clusters(examples, K)
    
    # K-means++ cluster 초기화
    clusters = init_clusters_advanced(examples, K)

    for i in range(T):
        examples_clusters_distance = compute_distance(examples, clusters)
        examples_nearest_cluster = find_nearest_cluster(examples_clusters_distance) # E Step
        predictions = examples_nearest_cluster
        clusters = update_clusters(examples, clusters, examples_nearest_cluster, K) # M Step
        examples_clusters_distance_cost = compute_cost(examples_clusters_distance)

        print("examples_clusters_distance_cost : ", examples_clusters_distance_cost)
    
    # Fill this

    return clusters, predictions

# K-means cluster 초기화
def init_clusters(examples, K):
    # # [examples] type : torch.Tensor / len : 60000 / shape : torch.Size([60000, 784])
    print("examples_type : ", type(examples))
    print("examples_len : ", len(examples))
    print("examples_shape : ", examples.shape)
    print("examples : ", examples)
    
    # examples 형태 변경 : 뭉탱이 텐서 -> 조각 텐서
    clusters = torch.unbind(examples, dim=0) # [clusters] type : tuple / len : 60000
    # print("clusters_type : ", type(clusters))
    # print("clusters_len : ", len(clusters))
    # print("clusters : ", clusters) # 이거 왜 print가 안되지???

    # 초기 10개 센트로이드 : K개 조각 cluster 추출 from sample
    clusters = random.sample(clusters, k=K) # [clusters] type : list / len : 10
    # print("clusters_type : ", type(clusters))
    # print("clusters_len : ", len(clusters))
    # print("clusters : ", clusters)

    # 초기 10개 센트로이드 : K개 조각 cluster -> K개 뭉탱이 cluster
    clusters = torch.stack(clusters, dim=0) # [clusters] type : torch.Tensor / len : 10 / shape : torch.Size([10, 784])
    # print("clusters_type : ", type(clusters))
    # print("clusters_len : ", len(clusters))
    # print("clusters_shape : ", clusters.shape)
    # print("clusters : ", clusters)
    return clusters

# K-means++ cluster 초기화
def init_clusters_advanced(examples, K):
    # # [examples] type : torch.Tensor / len : 60000 / shape : torch.Size([60000, 784])
    """ Implement K-means ++ algorithm """

    # examples 형태 변경 : 뭉탱이 텐서 -> 조각 텐서
    examples_unbind = torch.unbind(examples, dim=0) # [clusters] type : tuple / len : 60000
    # print("type(examples_unbind) : ", type(examples_unbind))

    # 초기 1개 센트로이드 : 무작위 샘플 1개 추출
    clusters = []
    clusters_idx = []
    init_example_idx = random.randint(0, len(examples_unbind))
    clusters_idx.append(init_example_idx)
    clusters.append(examples_unbind[init_example_idx])
    clusters_torch = torch.stack(clusters, dim=0) # [clusters] type : torch.Tensor / len : 1 / shape : torch.Size([1, 784])
    # print("type(clusters) : ", type(clusters_torch))
    # print("clusters.shape : ", clusters_torch.shape)

    # 초기 +9개 센트로이드 : 모든 examples 중에서 이전에 선택된 센트로이드와의 거리가 가장 먼 샘플 1개씩 추출
    while len(clusters_torch) < K:
        # cluster에 존재하는 example 제거
        # print("type(examples) : ", type(examples))
        # print("examples.shape : ", examples.shape)
        kmeans_examples = examples.numpy()
        # print("type(examples) : ", type(examples))
        # print("examples.shape : ", examples.shape)
        kmeans_examples = np.delete(kmeans_examples, clusters_idx, axis=0)
        # print("type(examples) : ", type(examples))
        # print("examples.shape : ", examples.shape)
        kmeans_examples = torch.from_numpy(kmeans_examples)
        # print("type(examples) : ", type(examples))
        # print("examples.shape : ", examples.shape)

        # kmeans_examples <-> clusters 거리 비교
        """
        kmeans_examples_clusters_distance = compute_distance(kmeans_examples, clusters_torch)
        # print("type(kmeans_examples_clusters_distance) : ", type(kmeans_examples_clusters_distance))
        # print("kmeans_examples_clusters_distance.shape : ", kmeans_examples_clusters_distance.shape)
        """
        examples_clusters_distance = compute_distance(examples, clusters_torch)

        # 각 kmeans_examples에서 clusters까지 거리 평균
        """
        kmeans_examples_clusters_distance_mean = kmeans_examples_clusters_distance.mean(dim=1)
        # print("type(kmeans_examples_clusters_distance_mean) : ", type(kmeans_examples_clusters_distance_mean))
        # print("kmeans_examples_clusters_distance_mean.shape : ", kmeans_examples_clusters_distance_mean.shape)
        """
        examples_clusters_distance_mean = examples_clusters_distance.mean(dim=1)

        # 모든 kmeans_examples 중에서 현재 clusters까지의 거리 평균이 가장 큰 index
        """
        kmeans_examples_clusters_distance_mean_max_idx = kmeans_examples_clusters_distance_mean.argmax(dim=0)
        # print("type(kmeans_examples_clusters_distance_mean_max_idx) : ", type(kmeans_examples_clusters_distance_mean_max_idx))
        # print("kmeans_examples_clusters_distance_mean_max_idx.shape : ", kmeans_examples_clusters_distance_mean_max_idx.shape)
        # print("kmeans_examples_clusters_distance_mean_max_idx : ", kmeans_examples_clusters_distance_mean_max_idx)
        
        examples_clusters_distance_mean_max_idx = examples_clusters_distance_mean.argmax(dim=0)
        print("examples_clusters_distance_mean_max_idx : ", examples_clusters_distance_mean_max_idx)
        """
        examples_clusters_distance_mean_list = examples_clusters_distance_mean.tolist()
        max_count = 0
        examples_clusters_distance_mean_max_idx = examples_clusters_distance_mean_list.index(sorted(examples_clusters_distance_mean, reverse=True)[max_count])
        while examples_clusters_distance_mean_list.index(sorted(examples_clusters_distance_mean, reverse=True)[max_count]) in clusters_idx:
            print("중복 idx : ", examples_clusters_distance_mean_max_idx)
            max_count += 1
            examples_clusters_distance_mean_max_idx = examples_clusters_distance_mean_list.index(sorted(examples_clusters_distance_mean, reverse=True)[max_count])
        print("examples_clusters_distance_mean_max_idx : ", examples_clusters_distance_mean_max_idx)

        # 새로운 센트로이드 추가
        """
        # print("kmeans_examples_clusters_distance_mean_max_idx : ", kmeans_examples_clusters_distance_mean_max_idx)
        clusters_idx.append(kmeans_examples_clusters_distance_mean_max_idx.item())
        # print("type(clusters_idx) : ", type(clusters_idx))
        # print("clusters_idx : ", clusters_idx)
        clusters.append(examples_unbind[kmeans_examples_clusters_distance_mean_max_idx])
        # print("type(clusters) : ", type(clusters))
        # print("clusters : ", clusters)
        
        clusters_idx.append(examples_clusters_distance_mean_max_idx.item())
        print("type(clusters_idx) : ", type(clusters_idx))
        print("clusters_idx : ", clusters_idx)
        """
        clusters_idx.append(examples_clusters_distance_mean_max_idx)
        # print("type(clusters_idx) : ", type(clusters_idx))
        # print("clusters_idx : ", clusters_idx)
        clusters.append(examples_unbind[examples_clusters_distance_mean_max_idx])
        clusters_torch = torch.stack(clusters, dim=0)

    return clusters

# cluster <-> examples 거리 계산
def compute_distance(examples, clusters):
    # # [examples] type : torch.Tensor / len : 60000 / shape : torch.Size([60000, 784])
    # print("examples_type : ", type(examples))
    # print("examples_len : ", len(examples))
    # print("examples_shape : ", examples.shape)
    # print("examples : ", examples)

    # # [clusters] type : torch.Tensor / len : 10 / shape : torch.Size([10, 784])
    # print("clusters_type : ", type(clusters))
    # print("clusters_len : ", len(clusters))
    # print("clusters_shape : ", clusters.shape)
    # print("clusters : ", clusters)
    
    # examples 형태 변경 : n c -> n 1 c
    examples = rearrange(examples, 'n c -> n 1 c') # [examples] type : torch.Tensor / len : 60000 / shape : torch.Size([60000, 1, 784])
    # print("examples_type : ", type(examples))
    # print("examples_len : ", len(examples))
    # print("examples_shape : ", examples.shape)
    # print("examples : ", examples)
    
    # 센트로이드 형태 변경 : k c -> 1 k c
    clusters = rearrange(clusters, 'k c -> 1 k c') # [clusters] type : torch.Tensor / len : 1 / shape : torch.Size([1, 10, 784])
    # print("clusters_type : ", type(clusters))
    # print("clusters_len : ", len(clusters))
    # print("clusters_shape : ", clusters.shape)
    # print("clusters : ", clusters)
    
    # cluster <-> examples 거리 계산
    distances = reduce((examples - clusters) ** 2, 'n k c -> n k', 'sum') # [distance] type : torch.Tensor / len : 60000 / shape : torch.Size([60000, 10])
    # # [(examples - clusters)] type : torch.Tensor, len : 60000, shape : torch.Size([60000, 10, 784])
    # print("(examples - clusters)_type : ", type((examples - clusters)))
    # print("(examples - clusters)_len : ", len((examples - clusters)))
    # print("(examples - clusters)_shape : ", (examples - clusters).shape)
    # print("(examples - clusters) : ", (examples - clusters))

    # # [distance] type : torch.Tensor / len : 60000 / shape : torch.Size([60000, 10]) ♣
    # print("distances_type : ", type(distances))
    # print("distances_len : ", len(distances))
    # print("distances_shape : ", distances.shape)
    # print("distances : ", distances)
    return distances

# 각 example에서 [cluster 10개와의 거리 중에서] 최소 거리 cluster 추출
def find_nearest_cluster(distances):
    # # [distances] type : torch.Tensor / len : 60000 / shape : torch.Size([60000, 10])
    # print("distances_type : ", type(distances))
    # print("distances_len : ", len(distances))
    # print("distances_shape : ", distances.shape)
    # print("distances : ", distances)
    cluster_ids = torch.argmin(distances, dim=-1) # [cluster_ids] type : torch.Tensor / len : 60000 / shape : torch.Size([60000])
    # print("cluster_ids_type : ", type(cluster_ids))
    # print("cluster_ids_len : ", len(cluster_ids))
    # print("cluster_ids_shape : ", cluster_ids.shape)
    # print("cluster_ids : ", cluster_ids)
    return cluster_ids

# cluster 업데이트
def update_clusters(examples, clusters, cluster_ids, K):
    for k in range(K):
        # 각 cluster에 할당된 examples idx 추출
        example_ids = torch.where(cluster_ids==k)[0] # [example_ids] type : torch.Tensor / len : 각 cluster에 할당된 examples 개수 / shape : torch.Size([각 cluster에 할당된 examples 개수])
        # print("example_ids_type : ", type(example_ids))
        # print("example_ids_len : ", len(example_ids))
        # print("example_ids_shape : ", example_ids.shape)
        # print("example_ids : ", example_ids)

        # 각 cluster에 할당된 examples 자체 추출 -> 각 cluster 내 examples의 평균값 추출 -> cluster 업데이트
        if len(example_ids) > 0:
            # [examples] type : torch.Tensor / len : 60000 / shape : torch.Size([60000, 784])
            # print("examples_type : ", type(examples))
            # print("examples_len : ", len(examples))
            # print("examples_shape : ", examples.shape)
            # print("examples : ", examples)
            cluster_examples = examples[example_ids, ...] # [cluster_examples] type : torch.Tensor / len : 각 cluster에 할당된 examples 개수 / shape : torch.Size([각 cluster에 할당된 examples 개수, 784])
            # print("cluster_examples_type : ", type(cluster_examples))
            # print("cluster_examples_len : ", len(cluster_examples))
            # print("cluster_examples_shape : ", cluster_examples.shape)
            # print("cluster_examples : ", cluster_examples)
            clusters[k] = reduce(cluster_examples, 'm c -> c', 'mean') # [clusters[k]] type : torch.Tensor / len : 784 / shape : torch.Size([784])
            # print("clusters[k]_type : ", type(clusters[k]))
            # print("clusters[k]_len : ", len(clusters[k]))
            # print("clusters[k]_shape : ", clusters[k].shape)
            # print("clusters[k] : ", clusters[k])

    # [clusters] type : torch.Tensor / len : 10 / shape : torch.Size([10, 784])
    # print("clusters_type : ", type(clusters))
    # print("clusters_len : ", len(clusters))
    # print("clusters_shape : ", clusters.shape)
    # print("clusters : ", clusters)
    return clusters

# 각 example에서 [cluster 10개와의 거리 중에서] 최소 거리 cluster까지의 거리 추출 -> 제곱 평균 = cost
def compute_cost(distances):
    # [distances] type : torch.Tensor / len : 60000 / shape : torch.Size([60000, 10])
    # print("distances_type : ", type(distances))
    # print("distances_len : ", len(distances))
    # print("distances_shape : ", distances.shape)
    # print("distances : ", distances)
    cost = reduce(distances, 'n m -> n', 'min') # [cost] type : torch.Tensor / len : 60000 / shape : torch.Size([60000])
    # print("cost_type : ", type(cost))
    # print("cost_len : ", len(cost))
    # print("cost_shape : ", cost.shape)
    # print("cost : ", cost)
    cost = reduce(cost ** 2, 'n -> 1', 'mean') # [cost] type : torch.Tensor / len : 1 / shape : torch.Size([1])
    # print("cost_type : ", type(cost))
    # print("cost_len : ", len(cost))
    # print("cost_shape : ", cost.shape)
    # print("cost : ", cost)
    return cost