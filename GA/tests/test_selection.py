import numpy as np
from GA.selection import select_parents


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
    Individuals with higher fitness should be selected more often *than lower-fitness
    individuals*, on a per-individual basis (not aggregated).
    """
    rng = np.random.RandomState(0)
    n_pop = 10

    # One-hot population; each row is unique and easily mapped to an index
    population = np.eye(n_pop, dtype=int)

    # Fitness increases with index; last one is best
    fitness = np.arange(n_pop, dtype=float)

    best_idx = np.argmax(fitness)   # n_pop - 1
    worst_idx = np.argmin(fitness)  # 0

    best_count = 0
    worst_count = 0

    # Run selection many times
    n_trials = 400
    for _ in range(n_trials):
        parents = select_parents(population, fitness, rng)

        # Convert one-hot rows back to indices
        selected_indices = parents.argmax(axis=1)

        best_count += np.sum(selected_indices == best_idx)
        worst_count += np.sum(selected_indices == worst_idx)

    # Best individual should be chosen more often than the worst individual
    assert best_count > worst_count

