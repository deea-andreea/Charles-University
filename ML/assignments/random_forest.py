#!/usr/bin/env python3
import argparse
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import math

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bagging", default=False, action="store_true", help="Perform bagging")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--feature_subsampling", default=1.0, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=73, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")


class Node:
    def __init__(self, left=None, right=None, value=None, feature=None, threshold=None):
        self.__left = left
        self.__right = right
        self.__value = value
        self.__feature = feature
        self.__threshold = threshold

    def is_leaf(self):
        return self.__value is not None

    def get_value(self):
        return self.__value

    def get_left(self):
        return self.__left

    def get_right(self):
        return self.__right

    def get_feature(self):
        return self.__feature

    def get_threshold(self):
        return self.__threshold


class DecisionTree:
    def __init__(self, max_depth, subsample_features_fn):
        self.__root = None
        self.__max_depth = max_depth
        self.__subsample_features_fn = subsample_features_fn
        self.n_features = None

    def entropy(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum([pi * np.log2(pi) for pi in p if pi > 0])

    def create_split(self, X, threshold):
        left_mask = X <= threshold
        right_mask = ~left_mask
        return left_mask, right_mask

    def can_split(self, depth, y):
        if self.__max_depth is not None and depth >= self.__max_depth:
            return False
        if len(np.unique(y)) == 1:
            return False
        return True

    def information_gain(self, X_col, y, threshold):
        parent_loss = self.entropy(y)
        left_mask, right_mask = self.create_split(X_col, threshold)
        y_left = y[left_mask]
        y_right = y[right_mask]
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        n = len(y)
        child_loss = (
            (len(y_left) / n) * self.entropy(y_left) +
            (len(y_right) / n) * self.entropy(y_right)
        )
        return parent_loss - child_loss

    def get_best_split(self, X, y, features):
        best_score = -1.0
        best_feature = None
        best_threshold = None
        n_samples = len(y)

        n_classes = int(np.max(y) + 1) if len(y) > 0 else 0
        if n_classes == 0:
            return None, None, 0.0

        parent_entropy = self.entropy(y)

        for feature in features:
            X_feat = X[:, feature]

            sort_indices = np.argsort(X_feat)
            X_feat_sorted = X_feat[sort_indices]
            y_sorted = y[sort_indices]

            count_right = np.bincount(y_sorted.astype(int), minlength=n_classes)
            count_left = np.zeros(n_classes, dtype=np.int64)

            n_right = n_samples
            n_left = 0

            for i in range(n_samples - 1):
                sample_class = y_sorted[i]

                count_right[sample_class] -= 1
                count_left[sample_class] += 1
                n_right -= 1
                n_left += 1

                if X_feat_sorted[i] != X_feat_sorted[i + 1]:
                    current_threshold = (X_feat_sorted[i] + X_feat_sorted[i + 1]) / 2

                    p_left = count_left / n_left
                    entropy_left = -np.sum(p_left * np.log2(p_left, where=p_left > 0))

                    p_right = count_right / n_right
                    entropy_right = -np.sum(p_right * np.log2(p_right, where=p_right > 0))

                    child_entropy = (n_left / n_samples) * entropy_left + (n_right / n_samples) * entropy_right

                    current_score = parent_entropy - child_entropy

                    if current_score > best_score:
                        best_score = current_score
                        best_feature = feature
                        best_threshold = current_threshold

        return best_feature, best_threshold, best_score

    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        most_common = np.argmax(np.bincount(y))

        if not self.can_split(depth, y):
            return Node(value=most_common)

        features = self.__subsample_features_fn(n_features)

        best_feat, best_thresh, best_score = self.get_best_split(X, y, features)

        if best_feat is None or best_score <= 1e-9:
            return Node(value=most_common)

        left_mask, right_mask = self.create_split(X[:, best_feat], best_thresh)

        left_child = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(left_child, right_child, None, best_feat, best_thresh)

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.__root = self.build_tree(X, y)

    def traverse_tree(self, x, node):
        if node.is_leaf():
            return node.get_value()
        if x[node.get_feature()] <= node.get_threshold():
            return self.traverse_tree(x, node.get_left())
        return self.traverse_tree(x, node.get_right())

    def predict(self, X):
        if self.__root is None:
            return np.zeros(len(X), dtype=int)
        return np.array([self.traverse_tree(x, self.__root) for x in X])


class RandomForest:
    def __init__(self, n_trees, max_depth, subsample_features_fn, use_bagging):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.subsample_features_fn = subsample_features_fn
        self.use_bagging = use_bagging
        self.trees = []


    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.n_trees):
            tree = DecisionTree(self.max_depth, self.subsample_features_fn)

            if self.use_bagging:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_train = X[indices]
                y_train = y[indices]
            else:
                X_train = X
                y_train = y

            tree.fit(X_train, y_train)
            self.trees.append(tree)

    def predict(self, X):
        all_predictions = np.vstack([tree.predict(X) for tree in self.trees])

        n_samples = X.shape[0]
        n_classes = int(np.max(all_predictions) + 1)
        result = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            votes = all_predictions[:, i]
            counts = np.bincount(votes.astype(int), minlength=n_classes)
            result[i] = np.argmax(counts)

        return result


def main(args: argparse.Namespace) -> tuple[float, float]:
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    generator_feature_subsampling = np.random.RandomState(args.seed)

    def subsample_features(number_of_features: int) -> np.ndarray:
        return np.sort(generator_feature_subsampling.choice(
            number_of_features, size=int(args.feature_subsampling * number_of_features), replace=False))



    forest = RandomForest(
        n_trees=args.trees,
        max_depth=args.max_depth,
        subsample_features_fn=subsample_features,
        use_bagging=args.bagging
    )

    np.random.seed(args.seed)
    forest.fit(train_data, train_target)

    train_pred = forest.predict(train_data)
    test_pred = forest.predict(test_data)

    train_accuracy = np.mean(train_pred == train_target)
    test_accuracy = np.mean(test_pred == test_target)


    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(main_args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))
