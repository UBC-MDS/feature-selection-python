import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from feature_selection import variance_thresholding

iris = pd.DataFrame(load_iris().data)


def test_1d_array_support():
    """
    Test with 1d array
    """
    result = variance_thresholding([1, 2, 3, 4, 5])
    assert np.array_equal(result, [0])


def test_2d_array_support():
    """
    Test with 2d array
    """
    result = variance_thresholding(
        [[1, 6, 0, 5], [1, 2, 4, 5], [1, 7, 8, 5]]
    )
    assert np.array_equal(result, [1, 2])


def test_df_support():
    """
    Test DataFrame support
    """
    iris_copy = pd.DataFrame.copy(iris)
    iris_copy['fake_num'] = np.zeros(iris_copy.shape[0])
    iris_copy['fake_categorical'] = 'abcde'

    result = variance_thresholding(iris_copy)
    assert np.array_equal(result, [0, 1, 2, 3, 5])


def test_invalid_data_exception():
    """
    Make sure the function throws errors for invalid input types
    """
    with pytest.raises(TypeError):
        assert variance_thresholding(0)

    with pytest.raises(TypeError):
        assert variance_thresholding('123')


def test_invalid_data_dim_exception():
    """
    Make sure the function throws errors for an input more than 2d
    """
    with pytest.raises(ValueError):
        assert variance_thresholding(
            [[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]
        )
