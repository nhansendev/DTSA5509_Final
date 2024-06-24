import numpy as np
import pandas as pd
from itertools import product, combinations
from copy import deepcopy
import time

import matplotlib.pyplot as plt
import matplotlib.patheffects as pathe
from matplotlib.lines import Line2D
from adjustText import adjust_text

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


def inspect_df(dataset):
    # Compact summary of dataFrame
    clen = max([len(c) for c in dataset.columns])
    for col in dataset.columns:
        vals, counts = np.unique(dataset[col], return_counts=True)
        if vals.dtype == np.object_:
            # Describe categorical data
            print(f"{col.ljust(clen)}: {vals.tolist()}, {counts.tolist()}")
        else:
            # Describe numerical data range
            print(f"{col.ljust(clen)}: {vals.min():.3f} to {vals.max():.3f}")


def categorical_to_columns(dataset, column):
    # Options like the different smoking statuses can be converted into n-1 one-hot columns
    uniques = pd.unique(dataset[column])
    for u in uniques[:-1]:
        dataset[column + "_" + u] = (dataset[column] == u).astype(int)
    dataset.drop(columns=column, inplace=True)


def categorical_to_numeric(dataset, column):
    # Categorical options like the different smoking statuses can be converted into numeric representations
    uniques = np.sort(pd.unique(dataset[column]))
    r_map = {u: str(i) for i, u in enumerate(uniques)}
    dataset.replace(r_map, inplace=True)
    dataset[column] = pd.to_numeric(dataset[column])
    return r_map


def standardize_columns(dataset, columns=None):
    if columns is None:
        columns = dataset.columns
    dataset[columns] = StandardScaler().fit_transform(dataset[columns].values)


def rebalance_by_class(dataset, random_seed=0):
    # The dataset is not balanced (many more non-stroke observations than stroke observations)
    # Re-sample from the stroke observations until we have 50% each stroke/non-stroke in the dataset
    subset = dataset[dataset["stroke"] == 1]
    dataset = dataset[dataset["stroke"] != 1]
    return pd.concat(
        [
            dataset,
            pd.DataFrame(
                resample(subset, n_samples=len(dataset), random_state=random_seed)
            ),
        ]
    )


def model_ensembles(models, N=range(2, 6)):
    # Start by getting all individual predictions
    preds = [
        models[i].best_test_model.predict(models[i].test_x) for i in range(len(models))
    ]

    # Shared by all models
    y_test = models[0].test_y

    # Separately track which combinations maximize the F1-Score and the Recall
    data_f1 = {n: None for n in N}
    data_recall = {n: None for n in N}
    for n in N:
        model_sets = list(combinations(range(len(models)), n))
        for s in model_sets:
            # Use majority-vote to determine if a stroke is predicted
            pred = np.mean([preds[i] for i in s], axis=0) >= 0.5
            f1 = f1_score(y_test, pred)
            rec = recall_score(y_test, pred)
            prec = precision_score(y_test, pred)
            if data_f1[n] is None or data_f1[n]["f1"] < f1:
                data_f1[n] = {
                    "f1": f1,
                    "rec": rec,
                    "prec": prec,
                    "idx": s,
                    "cm": confusion_matrix(y_test, pred),
                }
            if data_recall[n] is None or data_f1[n]["rec"] < rec:
                data_recall[n] = {
                    "f1": f1,
                    "rec": rec,
                    "prec": prec,
                    "idx": s,
                    "cm": confusion_matrix(y_test, pred),
                }

    df_f1 = pd.DataFrame(data_f1).T
    df_recall = pd.DataFrame(data_recall).T

    return df_f1, df_recall


def plot_ensembles(df_f1, df_recall, ref_values):
    # Plot the performance of the ensembles versus their size
    plt.plot(df_f1.index, df_f1["f1"], "b")
    plt.plot(df_f1.index, df_f1["rec"], "tab:orange")
    plt.plot(df_f1.index, df_f1["prec"], "g")
    plt.plot(df_recall.index, df_recall["f1"], "b:")
    plt.plot(df_recall.index, df_recall["rec"], ":", color="tab:orange")
    plt.plot(df_recall.index, df_recall["prec"], "g:")
    plt.hlines(ref_values, min(df_f1.index), max(df_f1.index), "k", "dashed")
    plt.grid()
    plt.title("Performance vs Ensemble Size")
    ax = plt.gca()
    lg = ax.legend(["F1-Score", "Recall", "Precision"])
    ax.add_artist(lg)
    plt.legend(
        handles=[
            Line2D([0], [0], color="k", label="Max F1-Score"),
            Line2D([0], [0], linestyle="dotted", color="k", label="Max Recall"),
            Line2D([0], [0], linestyle="dashed", color="k", label="Baseline"),
        ],
        loc="center left",
    )
    plt.ylabel("Best Scores")
    plt.xlabel("Ensemble Size")
    plt.gca().set_xticks(df_f1.index)
    plt.show()


def pair_plot(dataset, category_maps, figsize=(16, 12)):
    # Create a pair plot of the dataset
    axs = pd.plotting.scatter_matrix(
        dataset,
        figsize=figsize,
        range_padding=0.2,
        c=dataset["stroke"],
        cmap="cool",
        alpha=0.5,
    )
    N = len(dataset.columns) - 1
    # Change numeric labels to category names and Yes/No for binary
    for i, c in enumerate(dataset.columns):
        if c in category_maps.keys():
            axs[i, 0].set_yticks(list(range(len(category_maps[c].keys()))))
            axs[i, 0].set_yticklabels(category_maps[c].keys())
            axs[N, i].set_xticks(list(range(len(category_maps[c].keys()))))
            axs[N, i].set_xticklabels(category_maps[c].keys())
        elif i > 2:
            axs[i, 0].set_yticks([0, 1])
            axs[i, 0].set_yticklabels(["No", "Yes"])
            axs[N, i].set_xticks([0, 1])
            axs[N, i].set_xticklabels(["No", "Yes"])
    for ax in axs.flatten():
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha("right")
    plt.tight_layout()
    plt.show()


def PCA_plot(
    data,
    y_train,
    components,
    labels=None,
    importance=None,
    pairs=5,
    thr=0.2,
    cmax=5,
    legend_loc="lower left",
):
    # Visualize the results of the Principle Component Analysis
    r = pairs // cmax + 1
    c = min(cmax, pairs)
    fig, ax = plt.subplots(r, c, sharey="all")
    fig.set_size_inches(4 * c, 4 * r)
    axs = fig.axes

    for p in range(pairs):
        p1 = 2 * p
        p2 = 2 * p + 1

        xs = data[:, p1]
        ys = data[:, p2]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        tmp = axs[p].scatter(
            xs * scalex, ys * scaley, c=y_train, cmap="cool", s=3, alpha=0.5
        )

        # Make it easier to see the few stroke instances by re-plotting them on top
        idxs = y_train > 0
        axs[p].scatter(
            xs[idxs] * scalex,
            ys[idxs] * scaley,
            c=y_train[idxs],
            cmap="spring",
            s=3,
            alpha=0.5,
        )

        if p == 0:
            legend1 = axs[p].legend(
                tmp.legend_elements()[0],
                ["False", "True"],
                loc=legend_loc,
                title="Stroke",
            )
            axs[p].add_artist(legend1)

        texts = []
        # Plot arrows and labels for each component based on their coefficients in the PCA
        for i in range(components.shape[0]):
            if ((components[p1, i] ** 2 + components[p2, i] ** 2) ** 0.5) > thr:
                axs[p].arrow(
                    0,
                    0,
                    components[p1, i],
                    components[p2, i],
                    color="k",
                    alpha=0.75,
                    length_includes_head=True,
                    head_width=0.05,
                )
                texts.append(
                    axs[p].text(
                        components[p1, i],
                        components[p2, i],
                        labels[i],
                        color="w",
                        ha="center",
                        va="center",
                        path_effects=[pathe.withStroke(linewidth=2, foreground="k")],
                    )
                )
        axs[p].set_xlim(-1, 1)
        axs[p].set_ylim(-1, 1)
        axs[p].set_xlabel(f"PC{p1+1} ({importance[p1]:.1%})")
        axs[p].set_ylabel(f"PC{p2+1} ({importance[p2]:.1%})")

        adjust_text(texts, ax=axs[p], avoid_self=True)

    # Hide unused axes
    for i in range(1, len(axs) - p):
        axs[i + p].set_visible(False)


def plot_manual_search_grid(train_scores, test_scores, params, size=(16, 8)):
    # Visualize the results of performing grid search
    keys = list(params.keys())
    vals = list(params.values())

    fig, ax = plt.subplots(1, 2)
    axs = fig.axes
    fig.set_size_inches(size)

    axs[0].imshow(train_scores, interpolation="nearest", cmap="RdYlGn")
    axs[1].imshow(test_scores, interpolation="nearest", cmap="RdYlGn")

    # Add numeric labels to supplement the colors in the grid
    for i in range(train_scores.shape[0]):
        for j in range(train_scores.shape[1]):
            axs[0].text(j, i, f"{train_scores[i][j]:.3f}", ha="center", va="center")
            axs[1].text(j, i, f"{test_scores[i][j]:.3f}", ha="center", va="center")

    axs[0].set_ylabel(keys[0])
    axs[0].set_xlabel(keys[1])
    axs[1].set_ylabel(keys[0])
    axs[1].set_xlabel(keys[1])

    axs[0].set_xticks(np.arange(len(vals[1])), vals[1], rotation=45)
    axs[0].set_yticks(np.arange(len(vals[0])), vals[0])
    axs[1].set_xticks(np.arange(len(vals[1])), vals[1], rotation=45)
    axs[1].set_yticks(np.arange(len(vals[0])), vals[0])
    axs[0].set_title("Training F1-Scores")
    axs[1].set_title("Testing F1-Scores")
    plt.tight_layout()
    plt.show()


class GridSearch:
    # A custom implementation of grid search, including cross-validation and plotting
    def __init__(
        self, data, params, estimator, cv=3, rebalance=True, random_seed=0
    ) -> None:
        self.params = params
        self.estimator = estimator
        self.cv = cv  # cross-validation qty
        self.data = data
        self.rebalance = rebalance
        self.random_seed = random_seed

        train, test = self.simple_split(self.data)
        self.train_x, self.train_y = train
        self.test_x, self.test_y = test
        self.train_data, self.test_data = self.kfold_split(data)

        self.best_test_model = None
        self.train_score = None
        self.test_score = None

        self.keys = list(params.keys())
        self.val_lens = [len(v) for v in params.values()]
        vals = list(product(*list(params.values())))
        self.row_map = {str(p): i for i, p in enumerate(vals)}
        vals = [list(v) + [0.0, 0.0] for v in vals]
        cols = self.keys + ["train", "test"]
        self.data_df = pd.DataFrame(vals, columns=cols)

    def data_to_XY(self, data, target="stroke"):
        # Separate dataframe into target (Y) and everything else (X)
        return data.drop(columns=[target]), data[target]

    def kfold_split(self, data):
        # Split data for K-fold cross-validation
        KF = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_seed)
        train = []
        test = []
        for train_idx, test_idx in KF.split(data):
            if self.rebalance:
                train.append(
                    self.data_to_XY(
                        rebalance_by_class(data.iloc[train_idx], self.random_seed)
                    )
                )
            else:
                train.append(self.data_to_XY(data.iloc[train_idx]))
            test.append(self.data_to_XY(data.iloc[test_idx]))
        return train, test

    def simple_split(self, data, test_frac=0.2):
        # Generic test-train split
        data_train, data_test = train_test_split(
            data, test_size=test_frac, random_state=self.random_seed
        )
        if self.rebalance:
            data_train = rebalance_by_class(data_train, self.random_seed)
        return self.data_to_XY(data_train), self.data_to_XY(data_test)

    def get_scores(self, verbose=True):
        # Calculate training F1-Score as well as
        # testing F1-score, recall, and precision
        if self.best_test_model is None:
            self.get_best_test_model()

        score = self.test(self.best_test_model, self.train_x, self.train_y)
        self.train_score = score
        if verbose:
            print(f"Train F1 Score: {score:.3f}")

        pred = self.best_test_model.predict(self.test_x)
        self.test_score = f1_score(self.test_y, pred)
        if verbose:
            print(f"Test F1 Score: {self.test_score:.3f}")
        return {
            "train": self.train_score,
            "test": self.test_score,
            "precision": precision_score(self.test_y, pred),
            "recall": recall_score(self.test_y, pred),
        }

    def plot_confusion_matrix(self):
        # Show the test confusion matrix of the best model
        if self.best_test_model is None:
            self.get_best_test_model()

        pred = self.best_test_model.predict(self.test_x)

        conf_mat = confusion_matrix(self.test_y, pred)
        cm_plot = ConfusionMatrixDisplay(conf_mat)
        cm_plot.plot()
        plt.gcf().set_size_inches(4, 4)
        plt.title("Test Confusion Matrix")
        plt.show()

    def plot_search_grid(self, size=(16, 8)):
        # Show the result of the parameter grid search
        plot_manual_search_grid(*self.get_matrices(), self.params, size=size)

    def copy_estimator(self):
        return deepcopy(self.estimator)

    def get_best_test_model(self):
        # Train an instance of the estimator using the best parameter set
        model = self.copy_estimator()
        params = self.get_best_test_params()
        for k, v in params.items():
            model.__dict__[k] = v
        model.fit(self.train_x, self.train_y)
        self.best_test_model = model
        return model

    def fit(self, verbose=True):
        # Perform grid search
        st = time.time()
        dlen = len(self.data_df)
        for row in range(len(self.data_df)):
            if verbose and not row % (dlen // 10):
                print(f"Evaluating Parameter Set: {row+1}/{dlen} ({(row+1)/dlen:.1%})")
            est = self.copy_estimator()
            for k in self.keys:
                est.__dict__[k] = self.data_df.loc[row, k]
            for i in range(self.cv):
                X, Y = self.train_data[i]
                est.fit(X, Y)
                score = self.test(est, *self.train_data[i])
                self.data_df.at[row, "train"] += score / self.cv

                score = self.test(est, *self.test_data[i])
                self.data_df.at[row, "test"] += score / self.cv
        if verbose:
            print("Done")
        return time.time() - st

    def eval(self, name, model_dict, plot_confusion=True, verbose=True):
        # Perform grid search and report performance, with optional plots
        duration = self.fit(verbose)
        stats = self.get_scores(verbose)
        stats["time"] = duration
        stats["model"] = self
        model_dict[name] = stats
        if verbose:
            print("Best Parameters: ", self.get_best_test_params())
        if plot_confusion:
            self.plot_confusion_matrix()

    def test(self, model, X, Y):
        # Evaluate a given model's performance
        return f1_score(Y, model.predict(X))

    def get_matrices(self):
        # Get testing and training scores as numpy matrices (for grid plot)
        train_scores = np.array(self.data_df["train"]).reshape(*self.val_lens)
        test_scores = np.array(self.data_df["test"]).reshape(*self.val_lens)
        return train_scores, test_scores

    def get_best_test_params(self):
        row = self.data_df["test"].idxmax()
        return {k: self.data_df.loc[row, k] for k in self.keys}
