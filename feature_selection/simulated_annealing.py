class simulated_annealing:
	"""
	Feature selector that performs simmulated annealing to select features.

	The algorithm randomly chooses a set of features, trains on them, scores the model.
	Then the algorithm slightly modifies the chosen features randomly and tests to see
	if the model improves. If there is improvement, the newer model is kept, if not the
	algorithm tests to see if the worse model is still kept based on a acceptance
	probability that decreases as iterations continue and if the model performs worse.

	Parameters:
	-----------
	model : object
		Supervised learning model with a ```fit``` method

	c : int (default=1)
		Control rate of feature perturbation

	iterations : int (default=100)
		Number of iterations

	random_state : int (default=None)
		Random state of pseudo-random number generators

	Attributes:
	-----------
	features_ : numpy.array
		Boolean array of selected features

	Examples:
	---------
	>>> from feature_selection import simulated_annealing
	>>> X, y = make_friedman1(n_samples=100, n_features=10, random_state=123)
	>>> model = LinearRegression()
	>>> selector = simmulated_annealing(model, random_state=123)
	>>> selector = selector.fit(X, y)
	>>> selector.features_
	array([ False, False, True, True, False, False, False, False, False, False])
	"""

	def __init__(self, model, c=1, iterations=100, random_state=None):

	def fit(self, X, y):
	 """
        Trains simulated_annealing object to find significant features on a data set

        Parameters:
        -----------
        X : np.array
            feature training dataset
        Y : np.array
            target training dataset
        """


	def transform(self, X, y=None):
	"""
        Transforms data set to contain only significant features
        """
