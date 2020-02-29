def forward_select(model, X, y, min_features=None, max_features=10):
    """
    Select features using forward selection algorithm. It starts 
    as an empty model, and add the variable with the highest 
    improvement in the accuracy of the model. The process is 
    iteratively repeated and it stops when the remaining variables 
    doesn't improve the accuracy of the model.
    
    Parameters
    ----------
    model: sklearn.linear_model._base.LinearRegression
        the model to use for fitting the data
    X : numpy.ndarray
        training dataset
    y : numpy.ndarray
        test dataset
    min_features : int (default=None)
        number of minimum features to select
    max_features : int (default=10)
        number of maximum features to select

    Returns
    -------
    numpy ndarray
      A 1d array of indexes of the features that pass the threshold
      
    Examples
    --------
    >>> forward_select(LinearRegression(), train_data, test_data, max_features=5)
    array([1, 2, 5, 6, 8])
    """

