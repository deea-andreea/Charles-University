# Team:
# 28b0a1d9-ce24-4df2-bfa6-0b640bbbfeaf (Fiona)
# 71b3bcf5-b4b9-46dd-a800-29055f57c100 (Johannes)
# e5ae5a09-6a69-4329-b841-1a381fae3412 (Andreea)
#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--window_size", default=3, type=int,
                    help="Number of characters in the window left and right of the character to predict")


def pad_and_window(texts, window_size, return_windows=True):
    results = []
    for line in texts:
        if len(line) == 0:
            continue

        pad = max(window_size,
                  (2 * window_size + 1 - len(line)) // 2)
        padded = " " * pad + line + " " * pad

        for i in range(window_size, len(padded) - window_size):
            if return_windows:
                results.append(padded[i - window_size: i + window_size + 1])
            else:
                results.append(padded[i])
    return results


class Dataset:

    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()
        X_windows = pad_and_window(train.data.splitlines(), args.window_size, return_windows=True)
        y_chars = pad_and_window(train.target.splitlines(), args.window_size, return_windows=False)

        pipe = Pipeline([
            ("vectorizer", CountVectorizer(analyzer="char", ngram_range=(2, 4), max_features=13000)),
            ("clf", LogisticRegression(
                max_iter=300, random_state=42,
                solver="saga", C=7, n_jobs=-1, multi_class="multinomial", verbose=1
            )),
        ])

        model = pipe.fit(X_windows, y_chars)
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)


    else:
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as f:
            model = pickle.load(f)

        window_size = args.window_size
        half = window_size

        original_text = list(test.data)
        text = test.data.lower()
        text_chars = list(text)
        predicted = text_chars.copy()
        n = len(text_chars)

        for i, char in enumerate(original_text):
            if char.lower() in Dataset.LETTERS_NODIA:
                start = max(0, i - half)
                end = min(n, i + half + 1)
                window = text_chars[start:end]
                if len(window) < 2 * half + 1:
                    if start == 0:
                        window = [' '] * (2 * half + 1 - len(window)) + window
                    else:
                        window = window + [' '] * (2 * half + 1 - len(window))

                window_str = ''.join(window)

                pred_char = model.predict([window_str])[0]
                if pred_char.lower().translate(Dataset.DIA_TO_NODIA) != char.lower():
                    pred_char = char.lower()

                if char.isupper():
                    pred_char = pred_char.upper()

                predicted[i] = pred_char

            else:
                predicted[i] = char

        predictions = ''.join(predicted)
        output_file = "system_output.txt"

        with open(output_file, "w", encoding="utf-8") as out_file:
            out_file.write(predictions)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    result =main(main_args)
    # if result is not None:
    #     print(result)