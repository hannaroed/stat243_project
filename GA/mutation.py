import numpy as np

def mutate_population(population, mutation_rate, rng):
    """Flip bits with small probability."""

    mutation_mask = rng.rand(*population.shape) < mutation_rate

    return np.logical_xor(population, mutation_mask).astype(int)


