import numpy as np
from GA.fitness import evaluate_population_fitness


def generate_toy_data(rng, n_samples=80, n_features=5):
    """
    Create synthetic regression data where the first two features
    have real signal and the rest are noise.
    """
    X = rng.randn(n_samples, n_features)
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.randn(n_samples)
    return X, y


def test_fitness_output_shape_and_type():
    """
    Fitness output should have one value per chromosome.
    """
    rng = np.random.RandomState(0)
    X, y = generate_toy_data(rng)

    population = rng.randint(0, 2, size=(6, X.shape[1]))

    fitness = evaluate_population_fitness(
        population, X, y, cv=3, lambda_penalty=0.0, rng=rng
    )

    assert fitness.shape == (population.shape[0],)
    assert fitness.dtype.kind == "f"


def test_identical_chromosomes_same_fitness():
    """
    Identical chromosomes should obtain identical fitness values.
    """
    rng = np.random.RandomState(0)
    X, y = generate_toy_data(rng)

    chrom = np.array([1, 0, 0, 0, 0])
    population = np.vstack([chrom, chrom, chrom])

    fitness = evaluate_population_fitness(
        population, X, y, cv=3, lambda_penalty=0.0, rng=rng
    )

    assert np.allclose(fitness, fitness[0])


def test_signal_features_better_than_noise_only():
    """
    Chromosome selecting true signal features should outperform
    one that selects only noise features.
    """
    rng = np.random.RandomState(0)
    X, y = generate_toy_data(rng)

    chrom_good = np.array([1, 1, 0, 0, 0])
    chrom_bad = np.array([0, 0, 1, 1, 1])

    population = np.vstack([chrom_good, chrom_bad])

    fitness = evaluate_population_fitness(
        population, X, y, cv=3, lambda_penalty=0.0, rng=rng
    )

    assert fitness[0] > fitness[1]