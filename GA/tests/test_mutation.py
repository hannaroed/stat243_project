import numpy as np
from GA.mutation import mutate_population


def test_mutation_preserves_shape_and_binary_values():
    """
    Mutation should not change the population shape
    and should keep values binary.
    """
    rng = np.random.RandomState(0)
    population = rng.randint(0, 2, size=(10, 6))

    mutated = mutate_population(population, mutation_rate=0.2, rng=rng)

    assert mutated.shape == population.shape
    assert set(np.unique(mutated)).issubset({0, 1})


def test_mutation_rate_zero_no_change():
    """
    With mutation_rate = 0, population should be unchanged.
    """
    rng = np.random.RandomState(0)
    population = rng.randint(0, 2, size=(5, 4))

    mutated = mutate_population(population, mutation_rate=0.0, rng=rng)

    assert np.array_equal(mutated, population)


def test_mutation_rate_one_flip_all_bits():
    """
    With mutation_rate = 1, all bits should be flipped.
    """
    rng = np.random.RandomState(0)
    population = rng.randint(0, 2, size=(5, 4))

    mutated = mutate_population(population, mutation_rate=1.0, rng=rng)

    assert np.array_equal(mutated, 1 - population)