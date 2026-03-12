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
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")


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
        self.data = self.data.reshape([-1, 28*28]).astype(float)



def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:

        generator = np.random.RandomState(args.seed)

        np.random.seed(args.seed)
        train = Dataset()
        train.data = np.pad(train.data, [(0, 0), (0, 1)], constant_values=1)


        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
            train.data, train.target, test_size=0.1, random_state=args.seed)


        mlp = MLPClassifier(random_state=1,  verbose=1, hidden_layer_sizes=(512, 256, 128), alpha=0.00001)
        model = mlp.fit(train_data, train_target)

        pred = model.predict(test_data)
        accuracy = np.mean(pred == test_target)
        print(accuracy)

        mlp._optimizer = None
        for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        predictions = model.predict(test.data)
        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
