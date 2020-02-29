def variance_threshold_select(data, threshold = 0):
    """
    Select features above a certain threshold of variance
    Parameters
    ----------
    data : numpy ndarray, pandas DataFrame
      A numpy 2d array or a pandas DataFrame to select features from
    threshold : float, optional
      A variance threshold to filter features for
    Returns
    -------
    numpy ndarray
      A 1d array of indexes of the features that pass the threshold
    Examples
    --------
    >>> from feature_selection import variance_threshold_select
    >>> X = [[1, 6, 0, 5], [1, 2, 4, 5], [1, 7, 8, 5]]
    >>> variance_threshold_select(X)
    array([1, 2])
    """
