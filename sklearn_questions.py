"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `validate_data, check_is_fitted` functions
imported in this file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.


Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.multiclass import type_of_target


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        """Initialize the classifier.
        Parameters
        ----------
        n_neighbors : int, default=1
            Number of neighbors to use.
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Labels.

        Returns
        -------
        self : KNearestNeighbors
            Fitted estimator.
        """
        X, y = validate_data(
            self, X, y, ensure_2d=True, dtype="numeric", y_numeric=False
        )
        target_type = type_of_target(y)
        if target_type == "continuous":
            raise ValueError(
                "Unknown label type: continuous. "
                "This estimator is a classifier."
                " and requires discrete targets."
                )
        if self.n_neighbors is None:
            raise ValueError("n_neighbors must be set")

        self.n_neighbors = int(self.n_neighbors)
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")

        n_samples = X.shape[0]
        if self.n_neighbors > n_samples:
            raise ValueError(f"n_neighbors must be <= n_samples = {n_samples}")

        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        -------
        y_pred : ndarray of shape (n_test_samples,)
            Predicted labels.
        """
        check_is_fitted(self, ["X_train_", "y_train_"])

        X = validate_data(
            self, X, ensure_2d=True, dtype="numeric", reset=False
        )

        distances = pairwise_distances(X, self.X_train_)
        y_pred = np.empty(X.shape[0], dtype=self.y_train_.dtype)

        for i in range(X.shape[0]):
            idx_neighbors = np.argsort(distances[i])[: self.n_neighbors]
            neighbor_labels = self.y_train_[idx_neighbors]
            values, counts = np.unique(neighbor_labels, return_counts=True)
            y_pred[i] = values[np.argmax(counts)]

        return y_pred

    def score(self, X, y):
        """Compute accuracy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Accuracy.
        """
        y = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split."""
    def __init__(self, time_col="index"):
        """Initialize the splitter.

        Parameters
        ----------
        time_col : str, default="index"
            Column name to use as datetime information. If "index", use X.index.
        """
        self.time_col = time_col
    def _get_times(self, X):
        """Return datetime-like index/series used for splitting."""
        if self.time_col == "index":
            if not hasattr(X, "index"):
                raise ValueError("X must have an index to use time_col='index'")
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("X.index must be a DatetimeIndex when time_col='index'")
            return pd.DatetimeIndex(X.index)
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame when time_col is a column name")
        if self.time_col not in X.columns:
            raise ValueError(f"Column {self.time_col} not found in X")
        times = X[self.time_col]
        if not pd.api.types.is_datetime64_any_dtype(times):
            raise ValueError(f"Column {self.time_col} must have datetime dtype")
        return pd.DatetimeIndex(times)
    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations.

        Parameters
        ----------
        X : array-like
            Data to split.
        y : array-like, default=None
            Ignored, exists for compatibility.
        groups : array-like, default=None
            Ignored, exists for compatibility.
                    Returns
        -------
        n_splits : int
            Number of splits.
        """

        times = self._get_times(X)
        months = times.to_period("M")
        n_unique_months = len(pd.PeriodIndex(months).unique())
        return max(0, n_unique_months - 1)
    def split(self, X, y=None, groups=None):  # y should be optional
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like
            Data to split.
        y : array-like, default=None
            Ignored, exists for compatibility.
        groups : array-like, default=None
            Ignored, exists for compatibility.
        Yields
        -------
        train_indices : ndarray
            The training set indices for that split.
        test_indices : ndarray
            The testing set indices for that split.
        """
        times = self._get_times(X)
        months = times.to_period("M")
        unique_months = pd.PeriodIndex(months).unique().sort_values()
        for i in range(len(unique_months) - 1):
            m_train = unique_months[i]
            m_test = unique_months[i + 1]
            idx_train = np.where(months == m_train)[0].astype(int)
            idx_test = np.where(months == m_test)[0].astype(int)
            yield idx_train, idx_test
            