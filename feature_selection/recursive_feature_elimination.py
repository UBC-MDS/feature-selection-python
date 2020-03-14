from functools import reduce
from inspect import isfunction

import numpy as np
import pandas as pd


def recursive_feature_elimination(scorer, X, y, n_features_to_select=None):
    """
    Feature selector that implements recursive feature elimination

    Implements a greedy algorithm that iteratively fits and scores
    a scikit-learn classifier and eliminates features using
    a score-based metric.

    Parameters
    ----------
    scorer : function
        A custom user-supplied function that accepts X and y (as defined below)
        as input and returns the index of the column with the lowest weight.

    X : array-like of shape (n_samples, n_features)
        Training samples

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True values for X, used for training

    n_features_to_select : int or None (default=None)
        The number of features to be selected. If None, half the number
        of features are selected.

    Returns
    -------
    array of shape [n_features_to_select]
        List of column names or indices of non-eliminated features.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.linear_model import LinearRegression
    >>> from feature_selection import recursive_feature_elimination
    >>>
    >>> def scorer(X, y):
    >>>     model = LinearRegression()
    >>>     model.fit(X, y)
    >>>     return X.columns[model.coef_.argmin()]
    >>>
    >>> X, y = make_friedman1(n_samples=200, n_features=10, random_state=10)
    >>> result = recursive_feature_elimination(scorer, X, y,
    >>>                                        n_features_to_select=5)
    array([0, 1, 3, 4, 9])
    """
    # `scorer` must be a function
    if not isfunction(scorer):
        raise TypeError('scorer must be a function.')

    # Must be a numpy array or Pandas DataFrame
    if type(X) not in {pd.DataFrame, np.ndarray}:
        raise TypeError('X must be a a NumPy array or a Pandas DataFrame.')

    if len(X.shape) != 2:
        raise ValueError('X must be a 2-d array.')

    if type(y) not in {pd.DataFrame, np.ndarray}:
        raise TypeError('y must be a a NumPy array or a Pandas DataFrame.')

    if X.shape[0] != y.shape[0]:
        raise ValueError(f'X and y have inconsistent numbers of samples: '
                         '[{X.shape[0]}, {y.shape[0]}]')

    # Convert to Pandas DataFrame so that we can keep track of columns
    # by their column names. Pandas will assign column names 0, 1, etc.
    # Array indices are no good because they keep changing as we remove
    # columns.
    all_features = pd.DataFrame(X) if isinstance(X, np.ndarray) else X

    if n_features_to_select >= all_features.shape[1]:
        assert ValueError('n_features_to_select must be less then the number '
                          'of input features.')

    eliminated_features = []

    for i in range(1, X.shape[1]):
        # Remove currently eliminated features
        features_to_try = all_features.drop(columns=eliminated_features)

        # Get the next feature to remove
        feature_to_remove = scorer(features_to_try, y)
        eliminated_features.append(feature_to_remove)

        # If we have our target number of features, stop.
        if len(eliminated_features) + n_features_to_select >= X.shape[1]:
            break

    # Return a list of the features to keep
    eliminated_features = set(eliminated_features)

    kept_features = reduce(
        lambda acc, col: acc if col in eliminated_features else acc + [col],
        all_features.columns, [])

    return list(kept_features)
