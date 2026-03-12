#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization strength")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=92, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.



def main(args: argparse.Namespace) -> tuple[list[float], float, float]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset.
    data, target = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=args.seed)

    # TODO: Append a constant feature with value 1 to the end of all input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    constant_feature_array = np.ones((data.shape[0], 1))
    data = np.hstack((data, constant_feature_array))

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, target, test_size=args.test_size, random_state=args.seed)


    # Generate initial linear regression weights.
    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)
    weights_with_bias_zero = weights.copy()
    weights_with_bias_zero[-1] = 0

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        train_data_processed = train_data[permutation]
        train_target_processed = train_target[permutation]

        data_batches = np.array_split(train_data_processed, train_data.shape[0] / args.batch_size)
        train_batches = np.array_split(train_target_processed, train_target.shape[0] / args.batch_size)

        for i in range(len(data_batches)):
            data_batch = data_batches[i]
            train_batch = train_batches[i]

            ### (1 / |B|) * Σ_{i∈B} ((x_i^T w - t_i) * x_i) + λ * w

            unregularized_loss = (data_batch.T @ (data_batch @ weights - train_batch)) / data_batch.shape[0]
            weights_with_bias_zero = weights.copy()
            weights_with_bias_zero[-1] = 0
            l2_loss = args.l2 * weights_with_bias_zero
            gradient = unregularized_loss + l2_loss

            avg_gradient = gradient
            weights -= args.learning_rate * avg_gradient


        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `len(train_data)`.
        #
        # The gradient for the input example $(x_i, t_i)$ is
        # - $(x_i^T weights - t_i) x_i$ for the unregularized loss (1/2 MSE loss),
        # - $args.l2 * weights_with_bias_set_to_zero$ for the L2 regularization loss,
        #   where we set the bias to zero because the bias should not be regularized,
        # and the SGD update is
        #   weights = weights - args.learning_rate * gradient

        # TODO: Append current RMSE on train/test to `train_rmses`/`test_rmses`.

        train_pred = train_data @ weights
        train_error = train_pred - train_target
        train_rmses.append(np.sqrt(np.mean(train_error ** 2)))

        test_pred = test_data @ weights
        test_error = test_pred - test_target
        test_rmses.append(np.sqrt(np.mean(test_error ** 2)))


    # TODO: Compute into `explicit_rmse` test data RMSE when fitting
    # `sklearn.linear_model.LinearRegression` on `train_data` (ignoring `args.l2`).
    model = sklearn.linear_model.LinearRegression().fit(train_data, train_target)
    pred = model.predict(test_data)
    errors = pred - test_target
    mse = np.mean(errors ** 2)
    explicit_rmse = np.sqrt(mse)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, test_rmses[-1], explicit_rmse


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, sgd_rmse, explicit_rmse = main(main_args)
    print(sgd_rmse, explicit_rmse)
    print("Test RMSE: SGD {:.3f}, explicit {:.1f}".format(sgd_rmse, explicit_rmse))
    print("Learned weights:", *("{:.3f}".format(weight) for weight in weights[:12]), "...")
