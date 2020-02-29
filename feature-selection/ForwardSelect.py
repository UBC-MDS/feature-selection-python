class ForwardSelect:
    """
    The Forward Selection is an algorithm used to select features.
    It starts as an empty model, and add the variable with the 
    highest improvement in the accuracy of the model. The process 
    is iteratively repeated and it stops when the remaining variables 
    doesn't improve the accuracy of the model.

    Parameters
    ----------
    model : object
        the model to use for fitting the data
    min_features : int (default=None)
        number of minimum features to select
    max_features : int (default=10)
        number of maximum features to select

    Returns
    -------
    features_ : numpy ndarray
      Boolean array of selected features
      
    Examples
    --------
    >>> from feature-selection-python import ForwardSelect
	>>> lm = LinearRegression()
	>>> selector = ForwardSelect(lm, train_data, test_data, max_features=5)
	>>> my_selector = selector.fit(X, y)
    >>> my_selector.features_
    array([1, 2, 5, 6, 8])
    """

    def __init__(self, model, 
                 min_features=None, 
                 max_features=10):
        """
        Initialize a ForwardSelect object that use Forward algorithm 
        for Feature Selection.
        """
        self.max_features = max_features
        self.min_features = min_features
        self.model = model

    def fit(self, X, y):
	    """
        Trains ForwardSelect object to find significant features 
        on a data set.
        
        Parameters:
        -----------
        X : numpy.ndarray
            training dataset
        y : numpy.ndarray
            test dataset
        """	

	def transform(self, X, y=None):
	    """
        Transforms data set to contain only significant features.
        """
