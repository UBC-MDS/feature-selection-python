import numpy as np
import pandas as pd


def variance_thresholding(data, threshold=0):
    """
    Select features above a certain threshold of variance

    Parameters
    ----------
    data : numpy ndarray, pandas DataFrame, list
      A numpy array, a pandas DataFrame or list to select features from
    threshold : float, optional
      A variance threshold to filter features for

    Returns
    -------
    numpy ndarray
      A 1d array of indexes of the features that pass the threshold or
      are not numerical

    Examples
    --------
    >>> from feature_selection import variance_thresholding
    >>> X = [[1, 6, 0, 5], [1, 2, 4, 5], [1, 7, 8, 5]]
    >>> variance_thresholding(X)
    array([1, 2])
    """

    is_data_list = isinstance(data, list)
    is_data_df = isinstance(data, pd.DataFrame)
    is_data_np_array = isinstance(data, np.ndarray)

    if not (is_data_list or is_data_df or is_data_np_array):
        raise TypeError('Data is of an invalid type.')
    elif (is_data_np_array or is_data_list) and len(np.array(data).shape) > 2:
        raise ValueError(
            'Data is of an invalid shape. '
            'Please only pass in data of less than two dimensions.'
        )

    data_df = data if is_data_df else pd.DataFrame(data)
    variance_series = data_df.var(axis=0)

    # Get all non-numerical columns because .var only keeps numerical
    # columns and their variances
    non_num_columns = data_df.columns.difference(variance_series.index)

    # Get all numerical columns with variances above the threshold
    selected_num_columns = variance_series[
        variance_series > threshold
    ].index

    # Include all the non-numerical columns
    all_selected_columns = selected_num_columns.append(non_num_columns)

    # Transform the selected columns into column indexes
    selected_column_indexes = np.array(list(map(
        lambda x: data_df.columns.get_loc(x),
        all_selected_columns
    )))

    return np.sort(selected_column_indexes)
