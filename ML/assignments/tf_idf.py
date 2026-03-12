#!/usr/bin/env python3
import argparse
import lzma
import math
import os
import pickle
import re
import sys
import urllib.request
import warnings

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

# Deliberately ignore the liblinear-is-deprecated-for-multiclass-classification warning.
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore", "Using the 'liblinear' solver.*is deprecated.")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=False, action="store_true", help="Use IDF weights")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=79, type=int, help="Random seed")
parser.add_argument("--tf", default=False, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")


# For these and any other arguments you add, ReCodEx will keep your default value.


def compute_term_frequency(word_dictionary, bag_of_words):
    term_frequency_dictionary = {}
    length_of_bag_of_words = len(bag_of_words)

    for word, count in word_dictionary.items():
        term_frequency_dictionary[word] = count / float(length_of_bag_of_words)

    return term_frequency_dictionary

def compute_inverse_document_frequency(full_doc_list):
    idf_dict = {}
    length_of_doc_list = len(full_doc_list)

    idf_dict = dict.fromkeys(full_doc_list[0].keys(), 0)
    for word, value in idf_dict.items():
        idf_dict[word] = math.log(length_of_doc_list / (float(value) + 1))

    return idf_dict


class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names


def main(args: argparse.Namespace) -> float:
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)

    token_pattern = re.compile(r'\w+')
    word_counter = {}

    for doc in train_data:
        tokens = token_pattern.findall(doc)
        for token in tokens:
            word_counter[token] = word_counter.get(token, 0) + 1

    vocabulary = {word for word, count in word_counter.items() if count >= 2}

    word_to_index = {word: i for i, word in enumerate(vocabulary)}

    def build_matrix(text):
        data = np.zeros((len(text), len(vocabulary)), dtype=int)
        for i, doc in enumerate(text):
            for token in token_pattern.findall(doc):
                if token in word_to_index:
                    data[i, word_to_index[token]] += 1
        return data

    train_data = build_matrix(train_data)
    test_data = build_matrix(test_data)
    print(train_data.shape[1])


    if args.tf:
        train_sums = np.array(train_data.sum(axis=1)).flatten()
        train_sums[train_sums == 0] = 1
        train_data = train_data / train_sums[:, None]

        test_sums = np.array(test_data.sum(axis=1)).flatten()
        test_sums[test_sums == 0] = 1
        test_data = test_data / test_sums[:, None]
    else:
        train_data[train_data > 0] = 1
        test_data[test_data > 0] = 1

    if args.idf:
        N = train_data.shape[0]
        df = np.asarray((train_data > 0).sum(axis=0)).flatten()
        idf = np.log(N / (df + 1))

        train_data = np.multiply(train_data, idf)
        test_data = np.multiply(test_data, idf)


    model = sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000)
    model.fit(train_data, train_target)

    predictions = model.predict(test_data)

    f1_score = sklearn.metrics.f1_score(test_target, predictions, average="macro")

    return 100 * f1_score


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(main_args)
    print("F-1 score for TF={}, IDF={}: {:.1f}%".format(main_args.tf, main_args.idf, f1_score))
