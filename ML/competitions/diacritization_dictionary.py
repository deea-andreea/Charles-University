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

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict") # "fiction-train.txt"
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization_dictionary.model", type=str, help="Model path")
parser.add_argument("--window_size", default=4, type=int,
                    help="Number of characters in the window left and right of the character to predict")


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

class Dictionary:
    def __init__(self,
                 name="fiction-dictionary.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        self.variants = {}
        with open(name, "r", encoding="utf-8-sig") as dictionary_file:
            for line in dictionary_file:
                nodia_word, *variants = line.rstrip("\n").split()
                self.variants[nodia_word] = variants

def collect_training_samples(text_noaccent, text_withaccent, target_char, window):

    contexts = []
    labels = []
    n = len(text_noaccent)

    for pos, ch in enumerate(text_noaccent):
        if ch != target_char:
            continue

        left = max(0, pos - window)
        right = min(n, pos + window + 1)

        snippet = text_noaccent[left:pos] + text_noaccent[pos+1:right]

        contexts.append(snippet)
        labels.append(text_withaccent[pos])

    return contexts, labels

def trans_capita(source, target):
    if source.islower():
        return target.lower()
    if source.isupper():
        return target.upper()
    if source.istitle():
        return target.capitalize()
    return target

def extract_surrounding(line, idx, window_size):
    left = max(0, idx - window_size)
    right = min(len(line), idx + window_size + 1)
    return line[left:idx] + line[idx+1:right]

def get_variants(dict_obj, orig, low):
    return dict_obj.variants.get(orig) or dict_obj.variants.get(low)

def predict_char(model, ctx, target_char):
    try:
        probs = model.predict_proba([ctx])[0]
        classes = model.classes_
        idx = list(classes).index(target_char)
        return probs[idx]
    except ValueError:
        return 1e-9

def predict_full_word_no_dict(lword, row_lower, pos, submodels, window_size):
    out = []
    for i, ch in enumerate(lword):
        if ch not in Dataset.LETTERS_NODIA:
            out.append(ch)
            continue
        ctx = extract_surrounding(row_lower, pos + i, window_size)
        mdl = submodels[ch]
        out.append(mdl.predict([ctx])[0])
    return "".join(out)

def choose_best_variant(lword, variants, row_lower, pos, submodels, window_size):
    scores = []
    for var in variants:
        if len(var) != len(lword):
            scores.append(-np.inf)
            continue

        score = 0.0
        for i, dia_ch in enumerate(var):
            plain = lword[i]
            if plain not in Dataset.LETTERS_NODIA:
                continue

            ctx = extract_surrounding(row_lower, pos + i, window_size)
            model = submodels[plain]
            score += np.log(predict_char(model, ctx, dia_ch) + 1e-9)

        scores.append(score)

    return variants[np.argmax(scores)]


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()

        saved_models = {"models": {}}

        train_nodia = train.data.lower()
        train_dia   = train.target.lower()

        for ambig_char in sorted(list(set(Dataset.LETTERS_NODIA))):
            contexts, labels = collect_training_samples(
                                        train_nodia,
                                        train_dia,
                                        ambig_char,
                                        args.window_size)

            print(f"train a single classifier for {ambig_char}")

            clf = sklearn.pipeline.make_pipeline(
                sklearn.feature_extraction.text.CountVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5)
                ),
                sklearn.linear_model.LogisticRegression(
                    C=7.0,
                    solver="saga",
                    max_iter=200,
                    random_state=args.seed
                )
            )

            clf.fit(contexts, labels)
            saved_models["models"][ambig_char] = clf

        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(saved_models, model_file)

    else:
        test = Dataset(args.predict)
        dictionary = Dictionary()

        with lzma.open(args.model_path, "rb") as model_file:
            saved_models = pickle.load(model_file)

        text_orig_lines = test.data.split("\n")
        text_lower_lines = test.data.lower().split("\n")
        output_lines = []

        for line_idx, low_line in enumerate(text_lower_lines):
            orig_line = text_orig_lines[line_idx]

            low_words = low_line.split(" ")
            orig_words = orig_line.split(" ")

            rebuilt = []
            pos = 0

            for w_idx, low_w in enumerate(low_words):
                orig_w = orig_words[w_idx]

                if not low_w or not low_w.isalpha():
                    rebuilt.append(orig_w)
                    pos += len(low_w) + 1
                    continue

                variants = get_variants(dictionary, orig_w, low_w)

                if variants is None:
                    pred = predict_full_word_no_dict(low_w, low_line, pos, saved_models["models"], args.window_size)

                elif len(variants) == 1:
                    pred = variants[0]
                else:
                    pred = choose_best_variant(low_w, variants, low_line, pos, saved_models["models"], args.window_size)

                rebuilt.append(trans_capita(orig_w, pred))
                pos += len(low_w) + 1

            output_lines.append(" ".join(rebuilt))

        predictions = "\n".join(output_lines)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)