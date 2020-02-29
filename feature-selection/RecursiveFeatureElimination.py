class RecursiveFeatureElimination:
    """
    Feature selector that implements recursive feature elimination

    Implements a greedy algorithm that iteratively fits and scores
    a scikit-learn classifier and eliminates features using
    a score-based metric.

    Parameters
    ----------
    model : object
        Supervised learning model with a ```fit``` method

    n_features_to_select : int or None (default=None)
        The number of features to be selected. If None, half the number
        of features are selected.

    step : int or float, optional (default=1)
        If step >= 1, then number of features to remove at each iteration.
        If step < 1, percentage of fetures to remove at each iteration.


    Attributes
    ----------
    ranking_ : array of shape [n_features]
        List of ranked feature indices. That is, ranking_[i] corresponds to
        the position of the ith-ranked feature of the original data.


    Examples
    --------
    >>> from feature-selection import RecursiveFeatureElimination
    >>> model = LinearRegression()
    >>> selector = RecursiveFeatureElimination(model)
    >>> selector = selector.fit(X, y)
    >>> selector.ranking_
    array([ 5, 3, 1, 2])
    """

    def __init__(self, model, n_features_to_select=None, step=1, verbose=0):
        pass

    def fit(self, X, y):
        """
        Performs RFE

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X, used for training
        """
        pass

    def transform(self, X, y=None):
        """
        Transforms data set eliminating features determined in `fit`

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values
        """
        pass
