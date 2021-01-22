from ID3 import ID3, DT_Classify
import pandas as pd
import numpy as np
from collections import Counter


def get_trees_and_centroids(train_data: pd.DataFrame, N: int, p: float):
    n = len(train_data.index)
    trees_and_centroids = []
    for i in range(N):
        sub_train_data = train_data.sample(int(n * p))
        centroid = np.array(sub_train_data.mean(axis=0))
        features = list(train_data)
        features.remove('diagnosis')
        train_tree = ID3(sub_train_data, features, None, M=0)
        trees_and_centroids.append((train_tree, centroid))
    return trees_and_centroids


def get_sorted_distances(obj: pd.Series, trees_and_centroids: list):
    test_array = np.array(obj)
    test_array = test_array[1:]
    distances = []
    for j in range(len(trees_and_centroids)):
        centroid_j = trees_and_centroids[j][1]
        euclidean_dist = np.linalg.norm(test_array - centroid_j)
        distances.append((euclidean_dist, j))
    distances.sort(reverse=True)
    return distances


def KNN_Classify(obj: pd.Series, trees_and_centroids: list, distances: list, K: int):
    classifies = []
    for k in range(K):
        euclidean_dist, j = distances[k]
        train_tree = trees_and_centroids[j][0]
        c = DT_Classify(obj, train_tree)
        classifies.append(c)
    count = Counter(classifies)
    return count.most_common()[0][0]


def KNN(train_data: pd.DataFrame, test_data: pd.DataFrame, N: int, K: int, p: float):
    trees_and_centroids = get_trees_and_centroids(train_data, N, p)
    count_hits = 0
    count_misses = 0
    for i, row in test_data.iterrows():
        distances = get_sorted_distances(row, trees_and_centroids)  # [(dist_val, tree_index)]
        final_classify = KNN_Classify(row, trees_and_centroids, distances, K)
        actual_classify = row[0]
        if final_classify == actual_classify:
            count_hits += 1
        else:
            count_misses += 1

    res = count_hits / (count_hits + count_misses)

    return res


def main():
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    print(KNN(train_data, test_data, N=4, K=3, p=0.3))


if __name__ == "__main__":
    main()
