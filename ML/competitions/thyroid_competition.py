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
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, OneHotEncoder, StandardScaler, SplineTransformer, \
    PowerTransformer

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()
        train_data, test_data, train_target, test_target = model_selection.train_test_split(train.data,
                                                                                                    train.target,
                                                                                                    test_size=0.2,
                                                                                                    random_state=args.seed)

        columns_categ = []
        columns_num = []

        for i, col in enumerate(train.data.T):
            if np.all(np.floor(col) == col):
                columns_categ.append(i)
            else:
                columns_num.append(i)

        enc1 = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc2 = SplineTransformer(10)

        transformer = ColumnTransformer(
            [("categ", enc1, columns_categ),
            ("nr", enc2, columns_num)]
        )

        minmaxscaler = MinMaxScaler()
        poly = PolynomialFeatures()
        pipeline = Pipeline([
            ("transformer", transformer),
            ("scaler", minmaxscaler),
            ("poly_feature", poly),
            ("regression", LogisticRegression(random_state=args.seed, max_iter=10000))

        ])

        skf = StratifiedKFold(n_splits=5)
        parameters = {
            "poly_feature__degree": [1,2, 3],
            "regression__C": np.logspace(-8, 6, 10),
            "regression__solver": ["newton-cg"]
        }

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameters,
            cv=skf,
            scoring="accuracy"
        )

        model = grid_search.fit(train_data, train_target)
        pred = grid_search.best_estimator_.predict(test_data)

        test_accuracy = accuracy_score(pred, test_target)
        print(test_accuracy*100)

        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
