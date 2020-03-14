import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression

from feature_selection.recursive_feature_elimination import (
    recursive_feature_elimination)


def scorer(X, y):
    """
    Sample custom scorer that fits a model and returns
    an appropriate score for the feature selection problem.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Test samples

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True values for X, used for training

    Returns
    -------
    Index of feature column with the lowest weight.
    """
    model = LinearRegression()
    model.fit(X, y)
    return X.columns[model.coef_.argmin()]


def test_recursive_feature_elimination():
    """
    This test creates a dataset that has 5 features that are are used
    to compute `y`. The remaining 5 features are independent of `y`.
    This test should select the 5 feature columns used to compute `y`.
    """
    X, y = make_friedman1(n_samples=200, n_features=10, random_state=10)

    features = recursive_feature_elimination(scorer, X, y,
                                             n_features_to_select=4)

    assert features == [0, 1, 3, 4]

    # Test with n_features_to_select something other than 0.5 number
    # of total features to ensure logic to stop feature elimination
    # works correctly.
    features = recursive_feature_elimination(scorer, X, y,
                                             n_features_to_select=3)

    assert len(features) == 3

    # Retest with column names
    X, y = make_friedman1(n_samples=200, n_features=10, random_state=10)

    X = pd.DataFrame(X, columns=['zero', 'one', 'two', 'three', 'four',
                                 'five', 'six', 'seven', 'eight', 'nine'])

    features = recursive_feature_elimination(scorer, X, y,
                                             n_features_to_select=4)

    assert features == ['zero', 'one', 'three', 'four']


def test_rfe_parameter_scorer():
    other_params = [np.array([[0, 1], [2, 3]]), np.array([0, 0]), 1]

    with pytest.raises(TypeError):
        scorer = 10
        recursive_feature_elimination(scorer, *other_params)

    with pytest.raises(TypeError):
        scorer = 'text'
        recursive_feature_elimination(scorer, *other_params)

    with pytest.raises(TypeError):
        scorer = np.array([1, 2, 3])
        recursive_feature_elimination(scorer, *other_params)


def test_rfe_parameters():
    two_d_array = np.array([[0, 1], [2, 3]])
    three_d_array = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    other_params = [np.array([0, 0]), 1]

    # X must be an np.array
    with pytest.raises(TypeError):
        recursive_feature_elimination(scorer, 'a string!', *other_params)

    # X must not be 3-d
    with pytest.raises(ValueError):
        recursive_feature_elimination(scorer, three_d_array, *other_params)

    # X must not be 1-d either
    with pytest.raises(ValueError):
        recursive_feature_elimination(scorer, np.array([0]), *other_params)

    # y must be an np.array
    with pytest.raises(TypeError):
        recursive_feature_elimination(scorer, two_d_array, 'string!', 1)

    # y must not be 3-d
    with pytest.raises(ValueError):
        recursive_feature_elimination(scorer, two_d_array, three_d_array, 1)

    # X and y must have consistent number of samples
    with pytest.raises(ValueError):
        recursive_feature_elimination(scorer,
                                      two_d_array,
                                      np.array([0, 1, 2]), 1)
