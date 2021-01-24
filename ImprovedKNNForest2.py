from ID3 import ID3, DT_Classify
import pandas as pd
import numpy as np
from collections import Counter


def n_farthest_objects_by_indexes(train_data: pd.DataFrame, N: int):
    farthest_indexes = []
    n = len(train_data.index)
    distances = np.zeros((n, n))
    for i, row1 in train_data.iterrows():
        row1_array = np.array(row1)[1:]
        for j, row2 in train_data.iterrows():
            if i < j:
                row2_array = np.array(row2)[1:]
                distances[i][j] = np.linalg.norm(row1_array - row2_array)
    sum_list = list(distances.sum(axis=0))
    for i in range(len(train_data.index)):
        sum_list[i] = (sum_list[i], i)
    sum_list.sort(reverse=True)
    return sum_list[:N]


def n_closest_objects_by_indexes(root: pd.Series, train_data: pd.DataFrame, n: int):
    closest_objects = []
    root_array = np.array(root)[1:]
    for i, row in train_data.iterrows():
        row_array = np.array(row)[1:]
        euclidean_dist = np.linalg.norm(root_array - row_array)
        closest_objects.append((euclidean_dist, i))
    closest_objects.sort()
    closest_indexes = []
    for obj, i in closest_objects[1:n+1]:
        closest_indexes.append(i)
    return closest_indexes


def get_trees_and_centroids(train_data: pd.DataFrame, N: int, p: float):
    n = len(train_data.index)
    trees_and_centroids = []
    farthest_objects = n_farthest_objects_by_indexes(train_data, N)
    roots = []
    for obj in farthest_objects:
        roots.append(obj[1])
    for i in roots:
        root = train_data.iloc[i]
        closest_indexes = n_closest_objects_by_indexes(root, train_data, int(n * p))
        sub_train_data = train_data.iloc[closest_indexes]
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

    my_list = [1, 2, 3, 4, 5]
    p_list = [0.3, 0.4, 0.5, 0.6, 0.7]
    for p in p_list:
        for n in my_list:
            for k in my_list:
                if k <= n:
                    acc = KNN(train_data, test_data, N=n, K=k, p=p)
                    print("p={} N={} K={} acc={}".format(p, n, k, acc))


if __name__ == "__main__":
    main()
