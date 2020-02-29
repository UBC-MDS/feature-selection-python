## feature-selection 

![](https://github.com/UBC-MDS/feature-selection/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/feature-selection/branch/master/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/feature-selection) ![Release](https://github.com/UBC-MDS/feature-selection/workflows/Release/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/feature-selection/badge/?version=latest)](https://feature-selection.readthedocs.io/en/latest/?badge=latest)

Feature selection for scikit-learn estimators

### Overview:
If you have encountered a database with a myriad number of features, which could be messy to work on, a good idea is to approach this problem by selecting only some of these features for your model. Feature selection will reduce complexity, reduce the time when training an algorithm, and improve the accuracy of your model (if we select them wisely). However, this is not a trivial task. 

To help you out performing this task, we have created the **feature-selection** package in `python`.

If you are interested in a similar feature selection package for `R`, click [here](https://github.com/UBC-MDS/feature-selection-r).

### Feature description:
In this package, four functions are included to lead you with feature selection:

#### Forward
- Function that use the forward algorithm to select the number of features in a model. This iterative algorithm starts as an empty model, and add the variable with the highest improve in the accuracy of the model. The process then is iteratively repeated selecting the variables with the best improvement in the accuracy. This procedure stops when the remaining variables doesn't enhance the accuracy of the model.  

#### Backward
- TO DESCRIBE

#### Variance Thresholding  
Select features based on their variances. A threshold, typically a low one, would be set so that any feature with a variance lower than that would be filtered out. Since this algorithm only looks at features without their outputs, it could be used to do feature selection on data related to unsupervised learning.

#### Simulated Annealing  
Perform simmulated annealing to select features: randomly choose a set of features and determine model performance. Then slightly modify the chosen features randomly and test to see if the modified feature list has improved model performance. If there is improvement, the newer model is kept, if not, a test is performed to determine if the worse model is still kept based on a acceptance probability that decreases as iterations continue and how worse the newer model performs. The process is repeated for a set number of iterations.

### Installation:

```
pip install -i https://test.pypi.org/simple/ feature-selection
```

### Features
- TODO

### Dependencies

- TODO

### Usage

- TODO

### Documentation
The official documentation is hosted on Read the Docs: <https://feature-selection.readthedocs.io/en/latest/>

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
