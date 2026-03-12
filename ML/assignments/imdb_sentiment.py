#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import re
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
import sklearn.feature_extraction
import sklearn.metrics
import sklearn.model_selection
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="imdb_sentiment.model", type=str, help="Model path")
# TODO: Add other arguments (typically hyperparameters) as you need.


class Dataset:
    def __init__(self,
                 name="imdb_train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        self.data = []
        self.target = []
        with open(name) as f_imdb:
            for line in f_imdb:
                label, text = line.split("\t", 1)
                self.data.append(text)
                self.target.append(int(label))


def load_word_embeddings(
        name="imdb_embeddings.npz",
        url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
    if not os.path.exists(name):
        print("Downloading embeddings {}...".format(name), file=sys.stderr)
        urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
        os.rename("{}.tmp".format(name), name)

    with open(name, "rb") as f_emb:
        data = np.load(f_emb)
        words = data["words"]
        vectors = data["vectors"]
    embeddings = {word: vector for word, vector in zip(words, vectors)}
    return embeddings

def tokenize(text, embeddings):
    embedding_dim = len(next(iter(embeddings.values())))
    vectors = np.zeros((len(text), embedding_dim), dtype=np.float32)

    for i, token in enumerate(text):
        words = re.findall(r"\w+",token.lower())
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]
        word_vecs = [embeddings[w] for w in words if w in embeddings]
        word_vecs = np.stack(word_vecs)
        vectors[i] = word_vecs.mean(axis=0)

    return vectors



def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    word_embeddings = load_word_embeddings()

    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()

        print("Preprocessing dataset.", file=sys.stderr)
        train_as_vectors = tokenize(train.data, word_embeddings)

        print(type(train_as_vectors))

        train_x, validation_x, train_y, validation_y = sklearn.model_selection.train_test_split(
            train_as_vectors, train.target, test_size=0.25, random_state=args.seed)

        print("Training.", file=sys.stderr)
        # TODO: Train a model of your choice on the given data.
        model = sklearn.svm.LinearSVC()
        model.fit(train_x, train_y)

        print("Evaluation.", file=sys.stderr)
        validation_predictions = model.predict(validation_x)
        validation_accuracy = sklearn.metrics.accuracy_score(validation_y, validation_predictions)
        print("Validation accuracy {:.2f}%".format(100 * validation_accuracy))

        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        test_as_vectors = tokenize(test.data, word_embeddings)

        predictions = model.predict(test_as_vectors)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
