=====
Usage
=====

To use feature_selection in a project, you can import each of the functions individually.

.. code-block:: python

    from feature_selection import forward_selection
    from feature_selection import recursive_feature_elimination
    from feature_selection import simulated_annealing
    from feature_selection import variance_thresholding


You can also import the entire module first,

.. code-block:: python

    import feature_selection

and then use them like this:

.. code-block:: python

    feature_selection.forward_selection(...)

For details on the usage of specific functions, please refer to the `Index <./genindex.html>`_.
