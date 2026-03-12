#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--l2", default=1., type=float, help="L2 regularization factor")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.


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
    def __init__(self, max_depth, subsample_features_fn, l2):
        self.__root = None
        self.__max_depth = max_depth
        self.__subsample_features_fn = subsample_features_fn
        self.__l2 = l2
        self.n_features = None

    def leaf_value(self, g, h):
        return -np.sum(g) / (self.__l2 + np.sum(h))

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

    def entropy(self, y):
        raise NotImplementedError("Entropy is not used in GBDT implementation")

    def information_gain(self, X_col, y, threshold):
        raise NotImplementedError("Information gain is not used in GBDT implementation")

    def gb_gain(self, g, h, g_left, h_left, g_right, h_right):
        def score(G, H):
            return (G * G) / (self.__l2 + H)

        return 0.5 * (
                score(g_left, h_left) +
                score(g_right, h_right) -
                score(np.sum(g), np.sum(h))
        )

    def get_best_split(self, X, g, h, features):
        best_gain=-np.inf
        best_feature = None
        best_threshold = None

        n_samples = X.shape[0]

        for feature in features:
            X_feat = X[:, feature]

            sort_idx = np.argsort(X_feat)
            X_sorted = X_feat[sort_idx]

            g_sorted = g[sort_idx]
            h_sorted = h[sort_idx]

            g_right = np.sum(g)
            h_right = np.sum(h)
            g_left = 0.0
            h_left = 0.0

            for i in range(n_samples - 1):
                g_left += g_sorted[i]
                h_left += h_sorted[i]
                g_right -= g_sorted[i]
                h_right -= h_sorted[i]

                if X_sorted[i] == X_sorted[i + 1]:
                    continue
                threshold = (X_sorted[i] + X_sorted[i + 1]) / 2
                gain = self.gb_gain(g, h, g_left, h_left, g_right, h_right)

                if gain > best_gain:
                    best_gain=gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold, best_gain

    def build_tree(self, X, g, h, depth=0):
        n_samples, n_features = X.shape

        if self.__max_depth is not None and depth >= self.__max_depth or len(X) <= 1:
            return Node(value=self.leaf_value(g, h))

        features = self.__subsample_features_fn(n_features)

        best_feat, best_thresh, best_score = self.get_best_split(X, g, h, features)

        if best_feat is None or best_score <= 1e-9:
            return Node(value = self.leaf_value(g, h))

        left_mask, right_mask = self.create_split(X[:, best_feat], best_thresh)

        left_child = self.build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        right_child = self.build_tree(X[right_mask], g[right_mask], h[right_mask], depth + 1)
        return Node(left_child, right_child, None, best_feat, best_thresh)


    def fit(self, X, g, h):
        self.n_features = X.shape[1]
        self.__root = self.build_tree(X, g, h)

    def traverse_tree(self, x, node):
        if node.is_leaf():
            return node.get_value()
        if x[node.get_feature()] <= node.get_threshold():
            return self.traverse_tree(x, node.get_left())
        return self.traverse_tree(x, node.get_right())

    def predict(self, X):
        if self.__root is None:
            return np.zeros(len(X))
        return np.array([self.traverse_tree(x, self.__root) for x in X])


class GradientBoostingForest:
    def __init__(self, n_trees, max_depth, learning_rate, subsample_features_fn, l2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample_features_fn = subsample_features_fn
        self.l2 = l2
        self.trees = []

    def softmax(self, logits):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        n_classes = np.max(y) + 1
        logits = np.zeros((n_samples, n_classes))

        for i in range(self.n_trees):
            iteration_trees = []

            probabilities = self.softmax(logits)
            for c in range(n_classes):
                g = probabilities[:, c] - (y == c).astype(float)
                h = probabilities[:, c] * (1.0 - probabilities[:, c])

                tree = DecisionTree(self.max_depth, self.subsample_features_fn, self.l2)

                tree.fit(X, g, h)

                logits[:, c] += self.learning_rate*tree.predict(X)
                iteration_trees.append(tree)


            self.trees.append(iteration_trees)

    def predict_logits(self, X, n_trees):
        if n_trees is None:
            n_trees = len(self.trees)

        n_samples = X.shape[0]
        n_classes = len(self.trees[0])

        logits = np.zeros((n_samples, n_classes))

        for t in range(n_trees):
            for c, tree in enumerate(self.trees[t]):
                logits[:, c] += self.learning_rate * tree.predict(X)

        return logits

    def predict(self, X, n_trees=None):
        logits = self.predict_logits(X, n_trees)
        probs = self.softmax(logits)
        return np.argmax(probs, axis=1)

def main(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)


    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    classes = np.max(target) + 1

    model = GradientBoostingForest(
        n_trees=args.trees,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample_features_fn=lambda n: range(n),
        l2=args.l2
    )

    model.fit(train_data, train_target)

    train_accuracies = []
    test_accuracies = []

    for t in range(1, args.trees + 1):
        train_pred = model.predict(train_data, n_trees=t)
        test_pred = model.predict(test_data, n_trees=t)

        train_acc = sklearn.metrics.accuracy_score(train_target, train_pred)
        test_acc = sklearn.metrics.accuracy_score(test_target, test_pred)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    return [100 * acc for acc in train_accuracies], [100 * acc for acc in test_accuracies]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracies, test_accuracies = main(main_args)

    for i, (train_accuracy, test_accuracy) in enumerate(zip(train_accuracies, test_accuracies)):
        print("Using {} trees, train accuracy: {:.1f}%, test accuracy: {:.1f}%".format(
            i + 1, train_accuracy, test_accuracy))
