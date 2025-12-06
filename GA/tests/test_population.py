import numpy as np
from GA.population import initialize_population


def test_initialize_population_shape_and_binary_values():
    """
    Population should have shape (n_pop, n_features)
    and contain only binary values (0 or 1).
    """
    rng = np.random.RandomState(42)
    n_pop = 10
    n_features = 5

    population = initialize_population(n_pop, n_features, rng)

    assert population.shape == (n_pop, n_features)
    assert set(np.unique(population)).issubset({0, 1})


def test_initialize_population_reproducibility():
    """
    Using the same random seed should produce identical populations.
    """
    n_pop = 8
    n_features = 4

    rng1 = np.random.RandomState(123)
    rng2 = np.random.RandomState(123)

    pop1 = initialize_population(n_pop, n_features, rng1)
    pop2 = initialize_population(n_pop, n_features, rng2)

    assert np.array_equal(pop1, pop2)