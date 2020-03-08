import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression

def test_forward_selection():    
    """
    This test creates a dataset that has 5 features that are are used to compute `y`.
    The remaining 5 features are independent of `y`.
    This test should select the first 5 feature columns used to compute `y` more than the second set of 5 independent features.
    """
    data, target = make_friedman1(n_samples=200, n_features=10, random_state=10)
    features = forward_selection(scorer, data, target, min_features = 4, max_features=5)
    features = np.sort(features)
    assert (features is np.array([0, 1, 3, 4]) or features is np.array([0, 1, 2, 3, 4]))

    # Test min/max number of features working correctly
    features = forward_selection(scorer, data, target, min_features = 3, max_features=8)
    assert len(features) > 2
    assert len(features) < 9
    
    # Test ouput is correct type
    assert type(features[0]) is int 