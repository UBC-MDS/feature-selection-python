from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression

from feature_selection import recursive_feature_elimination


def scorer(X, y):
    """
    Sample custom scorer that fits a model and returns
    an appropriate score for the feature selection problem.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Test samples

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True values for X, used for training

    Returns
    -------
    Index of feature column with the lowest weight.
    """
    model = LinearRegression()
    model.fit(X, y)
    return X.columns[model.coef_.argmin()]


# This test creates a dataset that has 5 features that are are used to compute `y`.
# The remaining 5 features are independent of `y`.
# This test should select the 5 feature columns used to compute `y`.
def test_friedman():
    X, y = make_friedman1(n_samples=200, n_features=10, random_state=10)
    result = recursive_feature_elimination(scorer, X, y, n_features_to_select=5)
    assert result == [0, 1, 3, 4, 9]
