def recursive_feature_elimination(scorer, X, y, n_features_to_select=None, step=1, verbose=0):
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

    step : int or float, optional (default=1)
        If step >= 1, then number of features to remove at each iteration.
        If step < 1, percentage of fetures to remove at each iteration.

    Returns
    -------
    array of shape [n_features]
        List of ranked feature indices. That is, ranking_[i] corresponds to
        the position of the ith-ranked feature of the original data.

    Examples
    --------
    >>> from sklearn import datasets
    >>> from feature_selection import recursive_feature_elimination
    >>> iris = datasets.load_iris()
    >>> X, y = iris['data'], iris['target']
    >>> scorer = lambda x : return 0 # Real implementation should be to fit model and return score
    >>> recursive_feature_elimination(scorer, X, y, n_features_to_select=3)
    array([ 5, 3, 1, 2])
    """

    return []
