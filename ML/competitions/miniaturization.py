#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
import scipy
import sklearn.neural_network
from numpy import float32

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.0, type=float, help="Regularization strength")
parser.add_argument("--augment", default=True, action="store_true", help="Augment during training")
parser.add_argument("--augment_workers", default=16, type=int, help="Processes performing augmentation")
parser.add_argument("--dev", default=5000, type=float, help="Development fraction")
parser.add_argument("--epochs", default=15, type=int, help="Training epochs")
parser.add_argument("--hidden_layer", default=70, type=int, help="Hidden layer size")
parser.add_argument("--models", default=3, type=int, help="Model to train")
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")


class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in the range 0-255, the targets are numbers 0-9.
    """

    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28 * 28]).astype(float)


class MLPFullDistributionClassifier(sklearn.neural_network.MLPClassifier):
    class FullDistributionLabels:
        y_type_ = "multiclass"

        def fit(self, y):
            return self

        def transform(self, y):
            return y

        def inverse_transform(self, y):
            return np.argmax(y, axis=-1)

    def _validate_input(self, X, y, incremental, reset):
        X, y = sklearn.utils.validation.validate_data(
            self, X, y, multi_output=True, dtype=(np.float64, np.float32), reset=reset)
        if (not hasattr(self, "classes_")) or (not self.warm_start and not incremental):
            self._label_binarizer = self.FullDistributionLabels()
            self.classes_ = np.arange(y.shape[1])
        return X, y


def augment(x):
    x = x.reshape(28, 28)
    z = scipy.ndimage.zoom(x, (np.random.uniform(0.95, 1.05), np.random.uniform(0.95, 1.05)), order=1)
    z = scipy.ndimage.zoom(z, (28 / z.shape[0], 28 / z.shape[1]), order=1)

    z = scipy.ndimage.rotate(z, np.random.uniform(-7, 7), reshape=False, order=1)

    return z.reshape(-1).astype(np.float32)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        np.random.seed(args.seed)

        train = Dataset()
        train.data = train.data / 255.0
        train.data = train.data.reshape([-1, 28 * 28]).astype(np.float32)

        if args.dev:
            train.data, dev_data, train.target, dev_target = sklearn.model_selection.train_test_split(
                train.data, train.target, test_size=args.dev, random_state=args.seed
            )

        n_classes = 10
        train_target_onehot = sklearn.preprocessing.label_binarize(train.target, classes=np.arange(n_classes))

        models = [
            MLPFullDistributionClassifier(
                tol=1e-5,
                alpha=0.00001,
                hidden_layer_sizes=(300, 200),
                max_iter=50,
                learning_rate_init=0.001,
                batch_size=128,
                early_stopping=False,
                validation_fraction=0.1,
                n_iter_no_change=10,
                verbose=1
            )
            for _ in range(args.models)
        ]

        for idx, mlp in enumerate(models):
            print(f"Training MLP {idx + 1}/{len(models)}")
            mlp.fit(train.data, train_target_onehot)

            if args.augment:
                import multiprocessing
                pool = multiprocessing.Pool(args.augment_workers)

                # Disable early stopping for augmentation
                mlp.set_params(early_stopping=False, warm_start=True, max_iter=3)

                for epoch in range(1, min(args.epochs, 12)):  # Cap at 12 epochs
                    print(f"Augmenting data for epoch {epoch + 1}")
                    augmented_data = pool.map(augment, train.data)
                    mlp.fit(augmented_data, train_target_onehot)

            mlp._optimizer = mlp._best_coefs = mlp._best_intercepts = None

            for i in range(len(mlp.coefs_)):
                mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
            for i in range(len(mlp.intercepts_)):
                mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        if args.dev:
            dev_probs = np.mean([mlp.predict_proba(dev_data) for mlp in models], axis=0)
            dev_pred = np.argmax(dev_probs, axis=1)
            ensemble_acc = np.mean(dev_pred == dev_target)
            print(f"Ensemble development accuracy: {ensemble_acc:.4f}")

        compressed = []
        for m in models:
            coefs = [c.astype(np.float16) for c in m.coefs_]
            intercepts = [i.astype(np.float16) for i in m.intercepts_]
            compressed.append({"coefs": coefs, "intercepts": intercepts})

        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(compressed, model_file)

    else:
        test = Dataset(args.predict)
        test.data = test.data / 255.0
        with lzma.open(args.model_path, "rb") as model_file:
            compressed = pickle.load(model_file)

        models = []
        for m in compressed:
            mlp = MLPFullDistributionClassifier(hidden_layer_sizes=(300, 200))
            mlp.fit(np.zeros((1, 28 * 28)), np.zeros((1, 10)))
            mlp.coefs_ = [c.astype(np.float32) for c in m["coefs"]]  # Already float32
            mlp.intercepts_ = [i.astype(np.float32) for i in m["intercepts"]]
            models.append(mlp)

        probs = np.mean([mlp.predict_proba(test.data) for mlp in models], axis=0)
        predictions = np.argmax(probs, axis=1)
        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)