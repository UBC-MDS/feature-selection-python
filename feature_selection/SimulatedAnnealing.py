def SimulatedAnnealing(scorer, X, y, c=1, iterations=100, bools=False):
    """
    Feature selector that performs simmulated annealing to select features.

    The algorithm randomly chooses a set of features, trains on them, scores the model.
    Then the algorithm slightly modifies the chosen features randomly and tests to see
    if the model improves. If there is improvement, the newer model is kept, if not the 
    algorithm tests to see if the worse model is still kept based on a acceptance 
    probability that decreases as iterations continue and if the model performs worse.

    Parameters:
    -----------
    scorer : function
        Function that returns score of dataset
        
    X : np.array
        Feature training dataset
        
    Y : np.array
        Target training dataset

    c : int (default=1)
        Control rate of feature perturbation

    iterations : int (default=100)
        Number of iterations
        
    bools : bool (default=False)
        If true function returns array of boolean values

    Returns:
    --------
    numpy.array
        Array of selected features indicies
    """
    
    # Set mutate percentage
    mutate = 0.05
    
    # Obtain initial array of randomly selected features
    ftr_all = np.arange(0, X.shape[1])
    ftr_old = np.array([])
    while ftr_old.sum() == 0:
        ftr_old = np.random.binomial(1, 0.5, size=X.shape[1]).astype('bool')
    score_old = scorer(X[:,ftr_old], y)
    
    # Iterate through new versions of selected features
    for i in range(0, iterations):
        ftr_new = ftr_old.copy()
        ftr_mutate = np.array(random.sample(list(ftr_all),int(np.ceil(X.shape[1]*mutate))))
        for f in ftr_mutate:
            ftr_new[f] = not ftr_new[f]
        # Make sure new selected features has at least one feature
        if ftr_new.sum() != 0:
            score_new = scorer(X[:,ftr_new], y)
            if score_new < score_old:
                ftr_old = ftr_new
                score_old = score_new
            else:
                # Determine probability of acceptance
                p_accept = np.exp((-i/c)*((score_new-score_old)/score_old))
                if np.random.random() > p_accept:
                    pass
                else:
                    ftr_old = ftr_new
                    score_old = score_new

    if bools:
        return ftr_old
    else:
        return ftr_all[ftr_old]