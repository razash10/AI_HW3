from ID3 import ID3, DT_Classify
import pandas as pd
import numpy as np
from collections import Counter
import math


def increase_train_data(train_data: pd.DataFrame, target_n: int):
    current_n = len(train_data.index)
    new_train_data = pd.DataFrame(train_data)

    i = target_n - current_n  # number of rows to add
    while i > 0:
        for j, row in train_data.iterrows():
            new_train_data.loc[current_n] = row
            current_n += 1
            i -= 1
            if i == 0:
                break

    return new_train_data


def get_trees_and_centroids(train_data: pd.DataFrame, N: int, p: float):
    n = len(train_data.index)

    if N * p > 1:
        target_n = int(math.ceil(n * N * p))
        train_data = increase_train_data(train_data, target_n)

    # train_data = train_data.sample(frac=1, random_state=311177034)  # shuffle

    trees_and_centroids = []

    for i in range(N):
        sub_train_data = train_data[:int(math.ceil(n * p))]
        centroid = np.array(sub_train_data.mean(axis=0))
        features = list(train_data)
        features.remove('diagnosis')
        train_tree = ID3(sub_train_data, features, None, M=0)
        trees_and_centroids.append((train_tree, centroid))
        train_data = train_data.drop(list(sub_train_data.index.values))

    return trees_and_centroids


def get_sorted_distances(obj: pd.Series, trees_and_centroids: list):
    test_array = np.array(obj)
    test_array = test_array[1:]
    distances = []
    for j in range(len(trees_and_centroids)):
        centroid_j = trees_and_centroids[j][1]
        euclidean_dist = np.linalg.norm(test_array - centroid_j)
        distances.append((euclidean_dist, j))
    distances.sort()
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

    print(KNN(train_data, test_data, N=5, K=5, p=0.3))


if __name__ == "__main__":
    main()
