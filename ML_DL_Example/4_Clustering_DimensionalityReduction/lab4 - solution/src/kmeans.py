import torch
import random

from einops import rearrange, reduce


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
    # clusters = init_clusters(examples, K)
    clusters = init_clusters_advanced(examples, K)

    # Fill this
    distances = compute_distance(examples, clusters)
    predictions = find_nearest_cluster(distances)
    for _ in range(1, T):
        clusters = update_clusters(examples, clusters, predictions, K)
        distances = compute_distance(examples, clusters)
        predictions = find_nearest_cluster(distances)
    return clusters, predictions

def init_clusters(examples, K):
    clusters = torch.unbind(examples, dim=0)
    clusters = random.sample(clusters, k=K)
    clusters = torch.stack(clusters, dim=0)
    return clusters


def init_clusters_advanced(examples, K):
    """ Implement K-means ++ algorithm
    """
    clusters = torch.unbind(examples, dim=0)
    # Fill this
    clusters = random.sample(clusters, k=1)
    # For implementation of K-means++ pseudo code in lecture 7 slide page 42, 
    # we should remove selected example from examples. However, we won't do that in this implementation. 
    # Because, the distance between same points (cluster-exampe) is 0, this will lead the 0 probability.
    for i in range(1, K):
        distances = compute_distance(examples, clusters)
        distances = reduce(distances, 'n m -> n', 'min')
        total_distance = reduce(distances, 'n -> 1', 'sum')
        probs = distances / total_distance
        cluster = random.choices(examples, weights=probs, k=1)[0]
        clusters.append(cluster)
    clusters=torch.stack(clusters, dim=0)
    return clusters

def compute_distance(examples, clusters):
    examples = rearrange(examples, 'n c -> n 1 c')
    clusters = rearrange(clusters, 'k c -> 1 k c')
    distances = reduce((examples - clusters) ** 2, 'n k c -> n k', 'sum')
    return distances


def find_nearest_cluster(distances):
    cluster_ids = torch.argmin(distances, dim=-1)
    return cluster_ids


def update_clusters(examples, clusters, cluster_ids, K):
    for k in range(K):
        example_ids = torch.where(cluster_ids==k)[0]
        if len(example_ids) > 0:
            cluster_examples = examples[example_ids, ...]
            clusters[k] = reduce(cluster_examples, 'm c -> c', 'mean')
    return clusters
