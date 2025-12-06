import numpy as np
from project.selection import select_parents


def test_select_parents_shape_and_membership():
    """
    Selected parents must have the same shape as population
    and must be drawn from the original population.
    """
    rng = np.random.RandomState(0)
    n_pop = 12
    n_features = 4

    population = rng.randint(0, 2, size=(n_pop, n_features))
    fitness = rng.rand(n_pop)

    parents = select_parents(population, fitness, rng)

    assert parents.shape == population.shape

    for parent in parents:
        assert any(np.array_equal(parent, ind) for ind in population)


def test_select_parents_prefers_higher_fitness():
    """
    Individual with highest fitness should be selected most often.
    """
    rng = np.random.RandomState(0)
    n_pop = 10
    n_features = 3

    population = rng.randint(0, 2, size=(n_pop, n_features))
    fitness = np.zeros(n_pop)
    fitness[-1] = 10.0  # clearly best individual

    counts = np.zeros(n_pop, dtype=int)

    for _ in range(200):
        parents = select_parents(population, fitness, rng)
        for parent in parents:
            for idx, ind in enumerate(population):
                if np.array_equal(parent, ind):
                    counts[idx] += 1
                    break

    assert counts[-1] == counts.max()