import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

DEFAULT_PATH = "data"
DEFAULT_PATH_RAW = "data_raw"
DEFAULT_PATH_CLEAN = "data_clean"
DEFAULT_PATH_TEST = "data_test"
DEFAULT_PATH_OUT = "data_predict"
DEFAULT_EXTENSION = ".csv"


def import_data(path_name, has_test=False, raw=True):
    path_type = DEFAULT_PATH_RAW if raw else DEFAULT_PATH_CLEAN

    data_path = os.path.join(DEFAULT_PATH, path_type, path_name + DEFAULT_EXTENSION)
    data = pd.read_csv(data_path, delimiter=",")
    print(f"importing from: {data_path}")

    data_test = None
    if has_test:
        data_test_path = os.path.join(DEFAULT_PATH, DEFAULT_PATH_TEST, path_name + DEFAULT_EXTENSION)
        data_test = pd.read_csv(data_test_path, delimiter=",")

    return (data, data_test)


def export_data(data, path_name, predict=False):
    path_type = DEFAULT_PATH_OUT if predict else DEFAULT_PATH_CLEAN

    out_path = os.path.join(DEFAULT_PATH, path_type, path_name + DEFAULT_EXTENSION)
    data.to_csv(out_path)


def set_data_index(data, index):
    if index:
        data = data.set_index([index])
    return data


def seperate_data(data):
    data_numerical = data.select_dtypes(include=[np.number])
    data_categorical = data.select_dtypes(exclude=[np.number])

    return (data_numerical, data_categorical)


###################################################


def get_summary(data):
    summary = pd.DataFrame(data.dtypes, columns=["dtype"])
    summary["unique"] = data.nunique()
    summary["missing"] = data.isnull().sum()
    summary["duplicate"] = data.duplicated().sum()

    display(summary)


def plot_numerical_data(data, target_name):
    (data_numerical, _) = seperate_data(data)

    if len(data_numerical.columns) == 0:
        return

    number_cols = 3
    number_rows = math.ceil(len(data_numerical.columns) / number_cols)

    _, axes = plt.subplots(nrows=number_rows, ncols=number_cols, figsize=(20, 5 * number_rows))
    axes = axes.ravel()

    for ax in axes:
        ax.set_axis_off()

    for index, column in enumerate(data_numerical.columns):
        axes[index].set_axis_on()
        axes[index].set_title(str(column))

        data.groupby(target_name, observed=True)[column].plot.hist(bins=20, ax=axes[index], edgecolor="white")
        axes[index].legend()

    plt.tight_layout()
    plt.show()


def plot_categorical_data(data):
    (_, data_categorical) = seperate_data(data)

    if len(data_categorical.columns) == 0:
        return

    number_cols = 3
    number_rows = math.ceil(len(data_categorical.columns) / number_cols)

    _, axes = plt.subplots(nrows=number_rows, ncols=number_cols, figsize=(20, 5 * number_rows))
    axes = axes.ravel()

    for ax in axes:
        ax.set_axis_off()

    for index, column in enumerate(data_categorical.columns):
        axes[index].set_axis_on()
        axes[index].set_title(str(column))

        data[column].value_counts().sort_values().plot(kind="bar", ax=axes[index], xlabel="")

    plt.tight_layout()
    plt.show()


def print_correlation_matrix(data):
    (_, data_categorical) = seperate_data(data)

    # temporarily encode all categorical variables
    for column in data_categorical.columns:
        data = pd.get_dummies(data, columns=[column], drop_first=False)

    # create correlation matrix
    data_corr = data.corr()

    # fill diagonal and upper half of matrix with NaN
    mask = np.zeros_like(data_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    data_corr[mask] = np.nan

    # plot correlation matrix
    display(data_corr.style.background_gradient(cmap="coolwarm", axis=None).highlight_null(color="#f1f1f1").format(precision=2))
