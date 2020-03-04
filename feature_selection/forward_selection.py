def forward_selection(scorer, X, y, min_features=1, max_features=5):
    '''
    The Forward Selection is an algorithm used to select features.
    It starts as an empty model, and add the variable with the
    best improvement in the model. The process is iteratively 
    repeated and it stops when the remaining variables doesn't
    improve the accuracy of the model.

    Parameters
    ----------
    scorer : object
        function given by the user that returns the score. The
        forward_selection function, will chose the smaller score 
        from the scorer; in other words, the scorer should be 
        giving following the idea: "the smaller, the better"
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
      Numeric array of selected features

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import make_friedman1
    >>> data, target = make_friedman1(n_samples=200, n_features=15, random_state=0)
    >>> def my_scorer_fn2(X, y):
    >>>      lm = LinearRegression().fit(X, y)
    >>>      return 1 - lm.score(X, y)
    >>>
    >>> forward_selection(my_scorer_fn, data, target, max_features=7)
    array([0, 2, 3, 7, 9, 12, 13])
    '''
    ftr_ = []
    no_ftr_ = []
    scores_ = []
    step_ = []
    fn_score = []
    no_ftr_ = list(range(0, X.shape[1]))
    count = 1
    bandera = False
    X_new = []

    for j in range(0, max_features):
        for i in no_ftr_:
            X_new = X[:, ftr_ + [i] ]
            fn_score.append(scorer(X_new, y))

        # create data frame with the scores
        data = {'number': no_ftr_, 'fn_score':fn_score}
        df = pd.DataFrame(data)

        best_one = np.max(df.fn_score) 
        if len(ftr_) > 0:
            if best_one < np.max(scores_):
                if len(ftr_) > min_features:

                    bandera = True
                    break            

        x = df[df.fn_score == best_one].number
        scores_.append(best_one)
        ftr_.append(int(x))
        step_.append(count)
        no_ftr_.remove(int(x))

        data = {}
        fn_score = []
        count += 1

    return ftr_