import numpy as np

def crossover_population(parents, rate, rng):
    """Single point crossover entire population."""

    offspring = parents.copy()

    for i in range(0, len(parents), 2):
        if i+1 >= len(parents):
            break
        if rng.rand() < rate:
            cp = rng.randint(1, parents.shape[1])
            offspring[i, cp:], offspring[i+1, cp:] = \
                parents[i+1, cp:].copy(), parents[i, cp:].copy()
    return offspring
