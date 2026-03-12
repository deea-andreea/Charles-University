#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter of our NB classifier")
parser.add_argument("--naive_bayes_type", default="gaussian", choices=["gaussian", "multinomial", "bernoulli"])
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=72, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")


# If you add more arguments, ReCodEx will keep them with your default values.

def getPriors(X_train, y_train, labels):
    counts = {}
    for r in labels:
        data = X_train[np.where(y_train == r)[0], :]
        data = np.asarray(data)
        counts[r] = data.shape[0] / X_train.shape[0]
    return counts


def gaussianFormula(x, tple, alpha=None):
    mean, var = tple
    return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)


def estimateGaussianParameters(X_train, y_train, labels, alpha):
    d = [[() for _ in range(X_train.shape[1])] for _ in labels]
    for c in labels:
        Xc = X_train[y_train == c]
        for j in range(X_train.shape[1]):
            mean = Xc[:, j].mean()
            var = Xc[:, j].var() + alpha
            d[c][j] = (mean, var)
    return d


def estimateBernoulliParameters(X_train, y_train, labels, alpha):
    X_bin = (X_train >= 8).astype(int)
    d = [[0.0 for _ in range(X_train.shape[1])] for _ in labels]
    for c in labels:
        Xc = X_bin[y_train == c]
        N = Xc.shape[0]
        for j in range(X_train.shape[1]):
            count1 = np.sum(Xc[:, j])
            prob = (count1 + alpha) / (N + 2 * alpha)
            d[c][j] = prob
    return d




def estimateMultinomialParameters(X_train, y_train, labels, alpha):
    d = [[0.0 for _ in range(X_train.shape[1])] for _ in labels]
    for c in labels:
        Xc = X_train[y_train == c]
        N_c = np.sum(Xc)
        for j in range(X_train.shape[1]):
            count_j = np.sum(Xc[:, j])
            d[c][j] = (count_j + alpha) / (N_c + alpha * X_train.shape[1])
    return d


def main(args: argparse.Namespace) -> tuple[float, float]:
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    Y_pred = []

    cols = train_data.shape[1]
    labels = np.unique(train_target)

    priors = getPriors(train_data, train_target, labels)
    if args.naive_bayes_type == "gaussian":
        params = estimateGaussianParameters(train_data, train_target, labels, args.alpha)
    elif args.naive_bayes_type == "bernoulli":
        test_bin = (test_data >= 8).astype(int)
        params = estimateBernoulliParameters(train_data, train_target, labels, args.alpha)
    elif args.naive_bayes_type == "multinomial":
        params = estimateMultinomialParameters(train_data, train_target, labels, args.alpha)

    def logFeatureProb(x_val, feature_param, nb_type):
        if nb_type == "gaussian":
            return gaussianFormula(x_val, feature_param)
        elif nb_type == "bernoulli":
            p = feature_param
            return np.log(p) if x_val else np.log(1 - p)
        elif nb_type == "multinomial":
            return x_val * np.log(feature_param)
        else:
            raise NotImplementedError()

    Y_pred = []
    for i in range(test_data.shape[0]):
        x = test_data[i] if args.naive_bayes_type != "bernoulli" else test_bin[i]
        best_log, best_c = -np.inf, None
        for c in labels:
            log_prob = np.log(priors[c])
            for j in range(x.shape[0]):
                log_prob += logFeatureProb(x[j], params[c][j], args.naive_bayes_type)
            if log_prob > best_log:
                best_log, best_c = log_prob, c
        Y_pred.append(best_c)

    test_accuracy = np.mean(Y_pred == test_target)

    test_log_probability = 0.0
    for i in range(test_data.shape[0]):
        x = test_data[i] if args.naive_bayes_type != "bernoulli" else test_bin[i]
        c = test_target[i]
        log_prob = np.log(priors[c])

        for j in range(x.shape[0]):
            log_prob += logFeatureProb(x[j], params[c][j], args.naive_bayes_type)

        test_log_probability += log_prob

    return 100 * test_accuracy, test_log_probability


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy, test_log_probability = main(main_args)

    print("Test accuracy {:.2f}%, log probability {:.2f}".format(test_accuracy, test_log_probability))
