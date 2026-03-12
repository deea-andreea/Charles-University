#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")


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
    def __init__(self, min_to_split, max_depth, max_leaves):
        self.__root = None
        self.__max_depth = max_depth
        self.__max_leaves = max_leaves
        self.__min_to_split = min_to_split

    def gini(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

    def entropy(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum([p * np.log2(p) for p in p if p > 0])

    def create_split(self, X, threshold):
        left_idx = np.argwhere(X <= threshold).flatten()
        right_idx = np.argwhere(X > threshold).flatten()
        return left_idx, right_idx

    def can_split(self, depth, n_samples):
        if n_samples < self.__min_to_split:
            return False
        if self.__max_depth is not None and depth >= self.__max_depth:
            return False
        return True

    def information_gain(self, X_col, y, threshold, loss):
        if loss == 'gini':
            parent_loss = self.gini(y)
        else:
            parent_loss = self.entropy(y)
        left_idx, right_idx = self.create_split(X_col, threshold)
        n = len(y)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        if loss == 'gini':
            child_loss = (
                    (len(left_idx) / n) * self.gini(y[left_idx]) +
                    (len(right_idx) / n) * self.gini(y[right_idx])
            )
        else:
            child_loss = (
                    (len(left_idx) / n) * self.entropy(y[left_idx]) +
                    (len(right_idx) / n) * self.entropy(y[right_idx])
            )
        return parent_loss - child_loss

    def node_loss(self, y, criterion):
        return self.gini(y) if criterion == "gini" else self.entropy(y)

    def get_best_split(self, X, y, features, loss):
        best_score = -1
        best_feature = None
        best_threshold = None
        for feature in features:
            X_feat = X[:, feature]
            uniq = np.unique(X_feat)
            if len(uniq) < 2:
                continue
            thresholds = (uniq[:-1] + uniq[1:]) / 2
            for threshold in thresholds:
                score = self.information_gain(X_feat, y, threshold, loss)
                if score > best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold, best_score

    def build_tree(self, X, y, loss, depth=0):
        n_samples, n_features = X.shape
        most_common = np.argmax(np.bincount(y))

        if not self.can_split(depth, n_samples):
            return Node(value=most_common)

        features = np.arange(n_features)

        best_feat, best_thresh, best_score = self.get_best_split(X, y, features, loss)
        if best_feat is None or best_score <= 0:
            return Node(value=most_common)

        left_idx, right_idx = self.create_split(X[:, best_feat], best_thresh)
        left_child = self.build_tree(X[left_idx], y[left_idx], loss, depth + 1)
        right_child = self.build_tree(X[right_idx], y[right_idx], loss, depth + 1)
        return Node(left_child, right_child, None, best_feat, best_thresh)

    def build_tree_limited_leaves(self, X, y, loss):
        EPSILON = 1e-9

        leaves = []
        creation_counter = 0

        root_value = np.argmax(np.bincount(y))
        root = Node(value=root_value)
        leaves.append({"node": root, "idx": np.arange(len(y)), "depth": 0, "creation": creation_counter})
        creation_counter += 1

        while len(leaves) < self.__max_leaves:

            best_leaf_to_split = None
            best_feat_global = None
            best_thresh_global = None
            max_decrease_global = -float('inf')
            best_left_idx_global = None
            best_right_idx_global = None

            for leaf in leaves:
                idx = leaf["idx"]
                depth = leaf["depth"]
                n = len(idx)

                if not self.can_split(depth, n):
                    continue

                parent_criterion = self.node_loss(y[idx], loss)
                if parent_criterion <= EPSILON:
                    continue

                leaf_best_feat = None
                leaf_best_thresh = None
                leaf_best_decrease = -float('inf')
                leaf_best_left_idx = None
                leaf_best_right_idx = None

                for feat in range(X.shape[1]):
                    X_feat = X[idx, feat]
                    uniq = np.unique(X_feat)
                    if len(uniq) < 2:
                        continue
                    thresholds = (uniq[:-1] + uniq[1:]) / 2

                    for thresh in thresholds:
                        left_mask = X_feat <= thresh
                        right_mask = ~left_mask
                        left_idx_candidate = idx[left_mask]
                        right_idx_candidate = idx[right_mask]

                        if len(left_idx_candidate) == 0 or len(right_idx_candidate) == 0:
                            continue

                        left_criterion = self.node_loss(y[left_idx_candidate], loss)
                        right_criterion = self.node_loss(y[right_idx_candidate], loss)

                        weighted_children = (
                                len(left_idx_candidate) * left_criterion +
                                len(right_idx_candidate) * right_criterion
                        )
                        weighted_parent = n * parent_criterion

                        criterion_decrease = weighted_parent - weighted_children

                        if criterion_decrease > leaf_best_decrease:
                            leaf_best_decrease = criterion_decrease
                            leaf_best_feat = feat
                            leaf_best_thresh = thresh
                            leaf_best_left_idx = left_idx_candidate
                            leaf_best_right_idx = right_idx_candidate

                if leaf_best_feat is not None and leaf_best_decrease > EPSILON:

                    is_better = False

                    if leaf_best_decrease > max_decrease_global + EPSILON:
                        is_better = True

                    elif abs(leaf_best_decrease - max_decrease_global) < EPSILON:
                        if leaf["creation"] < best_leaf_to_split["creation"]:
                            is_better = True

                    if is_better:
                        max_decrease_global = leaf_best_decrease
                        best_leaf_to_split = leaf
                        best_feat_global = leaf_best_feat
                        best_thresh_global = leaf_best_thresh
                        best_left_idx_global = leaf_best_left_idx
                        best_right_idx_global = leaf_best_right_idx

            if best_leaf_to_split is None:
                break

            left_node = Node(value=np.argmax(np.bincount(y[best_left_idx_global])))
            right_node = Node(value=np.argmax(np.bincount(y[best_right_idx_global])))

            best_leaf_to_split["node"]._Node__left = left_node
            best_leaf_to_split["node"]._Node__right = right_node
            best_leaf_to_split["node"]._Node__value = None
            best_leaf_to_split["node"]._Node__feature = best_feat_global
            best_leaf_to_split["node"]._Node__threshold = best_thresh_global

            leaves.remove(best_leaf_to_split)

            leaves.append({
                "node": left_node,
                "idx": best_left_idx_global,
                "depth": best_leaf_to_split["depth"] + 1,
                "creation": creation_counter
            })
            creation_counter += 1
            leaves.append({
                "node": right_node,
                "idx": best_right_idx_global,
                "depth": best_leaf_to_split["depth"] + 1,
                "creation": creation_counter
            })
            creation_counter += 1

        return root

    def fit(self, X, y, loss):
        if self.__max_leaves is not None:
            self.__root = self.build_tree_limited_leaves(X, y, loss)
        else:
            self.__root = self.build_tree(X, y, loss)

    def traverse_tree(self, x, node):
        if node.is_leaf():
            return node.get_value()

        if x[node.get_feature()] <= node.get_threshold():
            return self.traverse_tree(x, node.get_left())
        return self.traverse_tree(x, node.get_right())

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.__root) for x in X])


def main(args: argparse.Namespace) -> tuple[float, float]:
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    tree = DecisionTree(
        min_to_split=args.min_to_split,
        max_depth=args.max_depth,
        max_leaves=args.max_leaves,
    )

    tree.fit(train_data, train_target, args.criterion)

    train_pred = tree.predict(train_data)
    test_pred = tree.predict(test_data)

    train_accuracy = np.mean(train_pred == train_target)
    test_accuracy = np.mean(test_pred == test_target)


    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(main_args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))