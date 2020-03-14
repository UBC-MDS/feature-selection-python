import numpy as np
import pytest

from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression

from feature_selection.simulated_annealing import simulated_annealing


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
    model = LinearRegression()
    model.fit(X, y)
    return 1 - model.score(X, y)


def test_simulated_annealing():
    """
    This test creates a dataset that has 5 features that
    are are used to compute `y`. The remaining 5 features are
    independent of `y`. This test should select the first 5
    feature columns used to compute `y` more than the second set
    of 5 independent features.
    """
    X, y = make_friedman1(n_samples=200, n_features=10, random_state=10)
    N = 10
    results = np.zeros((N, X.shape[1]))
    for n in range(0, N):
        results[n] = simulated_annealing(scorer, X, y, bools=True)
    assert results.sum(axis=0)[0] >= results.sum(axis=0)[5]
    assert results.sum(axis=0)[1] >= results.sum(axis=0)[6]
    # Omit feature 2 because weaker strength and harder to detec
    assert results.sum(axis=0)[3] >= results.sum(axis=0)[8]
    assert results.sum(axis=0)[4] >= results.sum(axis=0)[9]

    # Test output is non empty
    features = simulated_annealing(scorer, X, y)
    assert len(features) > 0
    features = simulated_annealing(scorer, X, y, bools=True)
    assert len(features) > 0

    # Test outputs are correct types
    features = simulated_annealing(scorer, X, y)
    assert isinstance(features[0], np.int64)
    features = simulated_annealing(scorer, X, y, bools=True)
    assert isinstance(features[0], np.bool_)


def test_sa_parameter_scorer():
    other_params = [np.array([[0, 1], [2, 3]]), np.array([0, 0])]

    with pytest.raises(TypeError):
        scorer = 10
        simulated_annealing(scorer, *other_params)

    with pytest.raises(TypeError):
        scorer = 'text'
        simulated_annealing(scorer, *other_params)

    with pytest.raises(TypeError):
        scorer = np.array([1, 2, 3])
        simulated_annealing(scorer, *other_params)


def test_sa_parameters():
    two_d_array = np.array([[0, 1], [2, 3]])
    three_d_array = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    other_params = [np.array([0, 0])]

    # X must be an np.array
    with pytest.raises(TypeError):
        simulated_annealing(scorer, 'a string!', *other_params)

    # X must not be 3-d
    with pytest.raises(ValueError):
        simulated_annealing(scorer, three_d_array, *other_params)

    # X must not be 1-d either
    with pytest.raises(ValueError):
        simulated_annealing(scorer, np.array([0]), *other_params)

    # y must be an np.array
    with pytest.raises(TypeError):
        simulated_annealing(scorer, two_d_array, 'string!', 1)

    # y must not be 3-d
    with pytest.raises(ValueError):
        simulated_annealing(scorer, two_d_array, three_d_array, 1)

    # X and y must have consistent number of samples
    with pytest.raises(ValueError):
        simulated_annealing(scorer, two_d_array, np.array([0, 1, 2]), 1)
