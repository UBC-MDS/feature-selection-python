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
    data, target = make_friedman1(n_samples=200, n_features=15, random_state=0)
    results = forward_selection(scorer, data, target, min_features = 5, max_features=10)
    assert len(results) > 5
    assert len(results) < 10
