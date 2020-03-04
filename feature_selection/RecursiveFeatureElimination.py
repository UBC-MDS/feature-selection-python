from functools import reduce

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
        A custom user-supplied function that creates and fits model
        and returns a score.

    X : array-like of shape (n_samples, n_features)
        Test samples

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
    >>> result = recursive_feature_elimination(scorer, X, y, n_features_to_select=5)
    array([0, 1, 3, 4, 9])
    """

    all_features = None

    if type(X) == np.ndarray:
        all_features = pd.DataFrame(X)

    if type(X) != pd.DataFrame:
        assert TypeError('X must be a Pandas DataFrame')

    eliminated_features = []

    for i in range(len(y)):
        # Remove currently eliminated features
        features_to_try = all_features.drop(columns=eliminated_features)

        # Get the next feature to remove
        feature_to_remove = scorer(features_to_try, y)
        eliminated_features.append(feature_to_remove)

        # If we have our target number of features, stop.
        if len(eliminated_features) >= n_features_to_select:
            break

    # Return a list of the features to keep
    eliminated_features = set(eliminated_features)
    kept_features = reduce(lambda acc, col: acc if col in eliminated_features else acc + [col],
                           all_features.columns, [])
    return list(kept_features)
