import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import dask
dask.config.set(scheduler="threads")
from dask import delayed, compute

def _chromosome_fitness(chrom, X, y, folds, lambda_penalty):
    """Helper function to compute fitness for a single chromosome."""

    selected = np.where(chrom == 1)[0]
    if len(selected) == 0:
        return -np.inf

    X_sub = X[:, selected]
    r2_scores = []

    for train_idx, test_idx in folds:
        X_train, X_test = X_sub[train_idx], X_sub[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2_scores.append(r2_score(y_test, preds))

    r2 = np.mean(r2_scores)
    penalty = lambda_penalty * (len(selected) / X.shape[1])
    return r2 - penalty


def evaluate_population_fitness(population, X, y, cv, lambda_penalty, rng):
    """
    Compute CV R2 for the whole population.
    Parallelized over chromosomes with Dask.
    How good is each chromosome? Converts chromosome --> fitness
    """

    n_pop = population.shape[0]

    # Deterministic, non-shuffled KFold
    kf = KFold(n_splits=cv)
    # Precompute folds once: splits depend only on n_samples, not on features
    folds = list(kf.split(X))

    # Build one delayed task per chromosome
    tasks = [
        delayed(_chromosome_fitness)(chrom, X, y, folds, lambda_penalty)
        for chrom in population
    ]

    # Run in parallel via Dask
    results = compute(*tasks)   # returns a tuple of fitness values
    fitness = np.array(results, dtype=float)

    return fitness
