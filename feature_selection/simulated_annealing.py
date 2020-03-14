from inspect import isfunction

import numpy as np
import pandas as pd
import random


def simulated_annealing(scorer, X, y, c=1, iterations=100, bools=False):
    """
    Feature selector that performs simmulated annealing to select features.

    Algorithm randomly chooses a set of features, trains on them, 
    scores the model. Then the algorithm slightly modifies the chosen 
    features randomly and tests to see if the model improves. If 
    there is improvement, the newer model is kept, if not the algorithm 
    tests to see if the worse model is still kept based on a acceptance
    probability that decreases as iterations continue and if the model 
    performs worse.

    Parameters:
    -----------
    scorer : function
        A custom user-supplied function that accepts X and y (as defined below)
        as input and returns the error of the datasets.

    X : np.array
        Feature training dataset

    y : np.array
        Target training dataset

    c : int (default=1)
        Control rate of feature perturbation

    iterations : int (default=100)
        Number of iterations

    bools : bool (default=False)
        If true function returns array of boolean values instead of 
        column indicies

    Returns:
    --------
    numpy.array
        Array of selected features indicies

    Examples:
    ---------
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.linear_model import LinearRegression
    >>> from feature_selection import simulated_annealing
    >>>
    >>> def scorer(X, y):
    >>>     model = LinearRegression()
    >>>     model.fit(X, y)
    >>>     return 1-lr.score(X, y)
    >>>
    >>> X, y = make_friedman1(n_samples=200, n_features=10, random_state=10)
    >>> simulated_annealing(scorer, X, y)
    array([ 0,  1,  3,  4,  5,  6,  7,  9, 10])
    """
    # `scorer` must be a function
    if not isfunction(scorer):
        raise TypeError('scorer must be a function.')

    # Must be a numpy array or Pandas DataFrame
    if type(X) not in {pd.DataFrame, np.ndarray}:
        raise TypeError('X must be a NumPy array or a Pandas DataFrame.')

    if len(X.shape) != 2:
        raise ValueError('X must be a 2-d array.')

    if type(y) not in {pd.DataFrame, np.ndarray}:
        raise TypeError('y must be a NumPy array or a Pandas DataFrame.')

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f'X and y have inconsistent numbers of samples: '
            '[{X.shape[0]}, {y.shape[0]}]')

    # Set mutate percentage
    mutate = 0.05

    # Obtain initial array of randomly selected features
    ftr_all = np.arange(0, X.shape[1])
    ftr_old = np.array([])
    while ftr_old.sum() == 0:
        ftr_old = np.random.binomial(1, 0.5, size=X.shape[1]).astype('bool')
    score_old = scorer(X[:, ftr_old], y)

    # Iterate through new versions of selected features
    for i in range(0, iterations):
        ftr_new = ftr_old.copy()
        ftr_mutate = np.array(
            random.sample(
                list(ftr_all), int(
                    np.ceil(
                        X.shape[1] * mutate))))
        for f in ftr_mutate:
            ftr_new[f] = not ftr_new[f]
        # Make sure new selected features has at least one feature
        if ftr_new.sum() != 0:
            score_new = scorer(X[:, ftr_new], y)
            if score_new < score_old:
                ftr_old = ftr_new
                score_old = score_new
            else:
                # Determine probability of acceptance
                p_accept = np.exp(
                    (-i / c) * ((score_new - score_old) / score_old))
                if np.random.random() > p_accept:
                    pass
                else:
                    ftr_old = ftr_new
                    score_old = score_new

    # Return either feature indicies or booleans
    if bools:
        return ftr_old
    else:
        return ftr_all[ftr_old]
