from feature_selection import variance_threshold_select
import numpy as np
import seaborn as sns
import pandas as pd
import pytest

iris = sns.load_dataset('iris')

def test_1d_array_support():
    result = variance_threshold_select([1, 2, 3, 4, 5])
    assert np.array_equal(result, [0])

def test_2d_array_support():
    result = variance_threshold_select(
        [[1, 6, 0, 5], [1, 2, 4, 5], [1, 7, 8, 5]]
    )
    assert np.array_equal(result, [1, 2])

def test_df_support():
    iris_copy = pd.DataFrame.copy(iris)
    iris_copy['fake_num'] = np.zeros(iris_copy.shape[0])
    iris_copy['fake_categorical'] = 'abcde'

    result = variance_threshold_select(iris_copy)
    assert np.array_equal(result, [0, 1, 2, 3, 4, 6])

def test_invalid_data_exception():
    with pytest.raises(Exception):
        assert variance_threshold_select(0)

    with pytest.raises(Exception):
        assert variance_threshold_select('123')

def test_invalid_data_dim_exception():
    with pytest.raises(Exception):
        assert variance_threshold_select(
            [[[1,2,3], [1,2,3]], [[1,2,3], [1,2,3]]]
        )
