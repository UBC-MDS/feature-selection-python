import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression

from feature_selection import forward_selection


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
    Error of scorer
    """
    lr = LinearRegression().fit(X, y)
    return 1 - lr.score(X, y)


def test_forward_selection():
    """
    This test creates a dataset that has 5 features that are are used to compute `y`.
    The remaining 5 features are independent of `y`.
    This test should select the first 5 feature columns used to compute `y` more than the second set of 5 independent features.
    """

    data, target = make_friedman1(
        n_samples=200, n_features=10, random_state=10)
    # Test feature selector working correctly
    results = forward_selection(
        scorer,
        data,
        target,
        min_features=4,
        max_features=4)
    results = np.sort(results)
    assert np.array_equal(results, np.array([0, 1, 3, 4]))

    # Test min/max number of features working correctly
    results = forward_selection(
        scorer,
        data,
        target,
        min_features=3,
        max_features=6)
    assert len(results) > 2
    assert len(results) < 7

    # Test ouput is correct type
    assert isinstance(results, list)
    assert isinstance(results[0], int)

    # test max_features should be greater or equal to min_features
    with pytest.raises(TypeError):
        forward_selection(scorer, data, target, min_features=5, max_features=3)

    # test min_features is positive, higher than zero
    with pytest.raises(TypeError):
        forward_selection(scorer, data, target, min_features=0, max_features=5)


def test_forward_selection_scorer():
    '''
    Tests when giving other value diferent from a function in scorer
    '''
    other_params = [np.array([[0, 1], [2, 3]]), 3, 5]

    with pytest.raises(TypeError):
        scorer = 10
        forward_selection(scorer, *other_params)

    with pytest.raises(TypeError):
        scorer = 'text'
        forward_selection(scorer, *other_params)

    with pytest.raises(TypeError):
        scorer = np.array([1, 2, 3])
        forward_selection(scorer, *other_params)


def test_forward_selection_datasets():
    '''
    Test for valid datasets in the forward function
    '''
    one_d_array = np.array([0, 1])
    two_d_array = np.array([[0, 1], [2, 3]])
    three_d_array = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    other_params = np.array([3, 5])

    # X must be an np.array
    with pytest.raises(TypeError):
        forward_selection(scorer, 'a string!', one_d_array, *other_params)

    # X must not be 3-d
    with pytest.raises(ValueError):
        forward_selection(scorer, three_d_array, one_d_array, *other_params)

    # X must not be 1-d either
    with pytest.raises(ValueError):
        forward_selection(scorer, one_d_array, one_d_array, *other_params)

    # y must be an np.array
    with pytest.raises(TypeError):
        forward_selection(scorer, two_d_array, 'string!', *other_params)

    # y must not be 2-d
    with pytest.raises(ValueError):
        forward_selection(scorer, two_d_array, two_d_array, *other_params)

    # y must not be 3-d
    with pytest.raises(ValueError):
        forward_selection(scorer, two_d_array, three_d_array, *other_params)

    # X and y must have consistent number of samples
    with pytest.raises(ValueError):
        forward_selection(scorer, two_d_array, np.array([0, 1, 2]), 1)
