import math
import string
import pandas as pd


class Node:
    """Contains the information of the node and another nodes of the Decision Tree."""

    def __init__(self, feature: string, objects: pd.DataFrame, default_c="", split_val=0.0):
        self.feature = feature
        self.objects = objects
        self.children = {}  # { value : Node }
        self.c = default_c
        self.entropy = self._entropy()
        self.split_val = split_val

    def _entropy(self):
        df = self.objects

        if len(df) == 0:
            return 0

        count_b = df[df.diagnosis == 'B'].shape[0]
        count_m = df[df.diagnosis == 'M'].shape[0]
        count_rows = df.shape[0]

        if count_b == 0 or count_m == 0:
            return 0

        prob_b = count_b / count_rows
        prob_m = count_m / count_rows

        log2_prob_b = math.log2(prob_b)
        log2_prob_m = math.log2(prob_m)

        res = -(prob_b * log2_prob_b + prob_m * log2_prob_m)

        return res


def ig_by_split_value(df: pd.DataFrame, feature: string, split_value: float) -> float:
    left_df = df[df[feature] <= split_value]
    right_df = df[df[feature] >= split_value]

    left_df_count = left_df.shape[0]
    right_df_count = right_df.shape[0]

    parent = Node(feature, df)
    left_son = Node("", left_df)
    right_son = Node("", right_df)
    count_objects = df.shape[0]

    left_ig = left_df_count * left_son.entropy / count_objects
    right_ig = right_df_count * right_son.entropy / count_objects

    ig = parent.entropy - left_ig - right_ig

    return ig


def max_info_gain(objects: pd.DataFrame, features: set) -> (string, float):
    max_ig = -1.0
    best_feature = ""
    best_split_value = 0

    for feature in features:
        df = objects.sort_values(by=feature)
        for i in range(df.shape[0]-1):
            var1 = df[feature].values[i]
            var2 = df[feature].values[i+1]
            split_value = (var1 + var2) / 2
            ig = ig_by_split_value(df, feature, split_value)
            if best_feature is None or ig > max_ig:
                max_ig = ig
                best_feature = feature
                best_split_value = split_value

    return best_feature, best_split_value


def ID3(examples: pd.DataFrame, features: set, default_c=None):
    if len(examples) == 0:
        return Node("", pd.DataFrame(), default_c)

    majority_class = examples.diagnosis.mode()[0]

    is_consistent_node = False
    if examples['diagnosis'].nunique() == 1:
        is_consistent_node = True

    if is_consistent_node or len(features) == 0:
        return Node("", pd.DataFrame(), majority_class)

    best_feature, split_value = max_info_gain(examples, features)

    sub_tree = Node(best_feature, examples, majority_class, split_value)

    examples.sort_values(by=best_feature)

    left_examples = examples[examples[best_feature] <= split_value]
    right_examples = examples[examples[best_feature] > split_value]
    left_son = ID3(left_examples, features, majority_class)
    right_son = ID3(right_examples, features, majority_class)
    sub_tree.children[0] = left_son
    sub_tree.children[1] = right_son

    return sub_tree


def DT_Classify(obj: pd.Series, tree: Node) -> string:
    if len(tree.children) == 0:
        return tree.c

    feature = tree.feature
    obj_value = obj[feature]

    left_son = tree.children[0]
    right_son = tree.children[1]

    if obj_value <= tree.split_val:
        return DT_Classify(obj, left_son)
    elif obj_value > tree.split_val:
        return DT_Classify(obj, right_son)
    else:
        assert True


def main():
    train_data = pd.read_csv("train.csv")
    features = set(train_data)
    features.remove('diagnosis')
    train_tree = ID3(train_data, features)

    test_data = pd.read_csv("test.csv")
    count_hits = 0
    count_misses = 0
    for i, row in test_data.iterrows():
        c_classified = DT_Classify(row, train_tree)
        c_actual = row[0]
        if c_classified == c_actual:
            count_hits += 1
        else:
            count_misses += 1

    res = count_hits / (count_hits + count_misses)

    print(res)


if __name__ == "__main__":
    main()
