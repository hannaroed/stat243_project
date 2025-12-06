import numpy as np
from project.crossover import crossover_population


def test_crossover_preserves_shape_and_binary():
    """
    Crossover should preserve population shape and keep values binary.
    """
    rng = np.random.RandomState(0)
    parents = rng.randint(0, 2, size=(10, 6))

    offspring = crossover_population(parents, rate=0.7, rng=rng)

    assert offspring.shape == parents.shape
    assert set(np.unique(offspring)).issubset({0, 1})


def test_crossover_rate_zero_no_change():
    """
    With crossover rate = 0, offspring should equal parents.
    """
    rng = np.random.RandomState(0)
    parents = rng.randint(0, 2, size=(8, 5))

    offspring = crossover_population(parents, rate=0.0, rng=rng)

    assert np.array_equal(offspring, parents)


def test_crossover_rate_one_changes_population():
    """
    With crossover rate = 1, at least some genes should change.
    """
    rng = np.random.RandomState(0)
    parents = rng.randint(0, 2, size=(10, 6))

    offspring = crossover_population(parents, rate=1.0, rng=rng)

    assert not np.array_equal(offspring, parents)