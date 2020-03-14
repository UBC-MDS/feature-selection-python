## feature-selection

![](https://github.com/UBC-MDS/feature-selection-python/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/feature-selection-python/branch/master/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/feature-selection-python) ![Release](https://github.com/UBC-MDS/feature-selection-python/workflows/Release/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/feature-selection/badge/?version=latest)](https://feature-selection-python.readthedocs.io/en/latest/?badge=latest)

Feature selection for machine learning models

### Overview:
If you have encountered a database with a myriad number of features, which could be messy to work on, a good idea is to approach this problem by selecting only some of these features for your model. Feature selection will reduce complexity, reduce the time when training an algorithm, and improve the accuracy of your model (if we select them wisely). However, this is not a trivial task.

To help you out performing this task, we have created the **feature-selection** package in `python`.

If you are interested in a similar feature selection package for `R`, click [here](https://github.com/UBC-MDS/feature-selection-r).

### Features:
In this package, four functions are included to lead you with feature selection:

- `forward_selection` - Function that use the Forward Selection algorithm to select the number of features in a model. This iterative algorithm starts as an empty model, and add the variable with the highest improve in the accuracy of the model. The process then is iteratively repeated selecting the variables with the best improvement in the accuracy. This procedure stops when the remaining variables doesn't enhance the accuracy of the model.  

- `recursive_feature_elimination` - Iteratively fit and score an estimator for greedy feature elimination.

- `simulated_annealing` - Perform simmulated annealing to select features: randomly choose a set of features and determine model performance. Then slightly modify the chosen features randomly and test to see if the modified feature list has improved model performance. If there is improvement, the newer model is kept, if not, a test is performed to determine if the worse model is still kept based on a acceptance probability that decreases as iterations continue and how worse the newer model performs. The process is repeated for a set number of iterations.

- `variance_threshold_select` - Select features based on their variances. A threshold, typically a low one, would be set so that any feature with a variance lower than that would be filtered out. Since this algorithm only looks at features without their outputs, it could be used to do feature selection on data related to unsupervised learning.

### Existing Ecosystems:
Some of the above features already exsist within the Python ecosystem:

- [Forward Selection] = None

- [Recursive Feature Elimination](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)

- [Variance Threshold](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html)

- [Simulated Annealing] = None


### Installation:

```
pip install -i https://test.pypi.org/simple/ feature-selection
```

### Dependencies

- [python 3.7.5](https://www.python.org/downloads/release/python-375/)
- [numpy 1.17.4](https://numpy.org/)
- [pandas 0.25.3](https://pandas.pydata.org/getpandas.html)

### Usage

To guide you with an example of how to use this package, we would use the [Friedman dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html).

Load libraries and dataset:
```
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_friedman1
X, Y = make_friedman1(n_samples=200, n_features=15, random_state=0)
```

Use of feature selection functions:

- forward_selection
```
# create a 'scorer'
def scorer(X, y):
    lm = LinearRegression().fit(X, y)
    return 1 - lm.score(X, y)

# use function
from feature_selection.forward_selection import forward_selection
forward_selection(scorer, X, Y, 3, 6)
```
output: [3, 1, 0, 4]

- recursive_feature_elimination
```
# create a 'scorer'
def scorer(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return X.columns[model.coef_.argmin()]

# use function
from feature_selection.recursive_feature_elimination import recursive_feature_elimination
recursive_feature_elimination(scorer, X, y, n_features_to_select=5)
```
output: [0, 1, 2, 10, 14]

- simulated_annealing
```
# create a 'scorer'
def scorer(X, y):
    lm = LinearRegression().fit(X, y)
    return 1 - lm.score(X, y)

# use function
from feature_selection.simulated_annealing import simulated_annealing
simulated_annealing(scorer, X, y)
```
output: array([ 1,  2,  3,  6,  7,  9, 10, 13])

- simulated_annealing  
*note: for this function we would use different data.*
```
X = [[1,6,0,5],[1,2,4,5],[1,7,8,5]]

# use function
feature_selection.variance_threshold_select import variance_threshold_select
variance_threshold_select(X)
```
output: array([1, 2])

### Documentation
The official documentation is hosted on Read the Docs: <https://feature-selection.readthedocs.io/en/latest/> 

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
