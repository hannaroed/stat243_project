import numpy as numpy

def initialize_population(n_pop, n_features, rng):
    """Generate Initial Random Chromosomes"""


    return rng.randint(0, 2, size=(n_pop, n_features))

    # draws random integers in interval [0, 1]
    # creates a matrix with shape (n_pop, n_features)
    # this is the matrix of P chromosomes 
    # rows are chrosomes 
    # each column is a feature 

