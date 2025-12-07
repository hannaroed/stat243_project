import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import dask
dask.config.set(scheduler="threads")
from dask import delayed, compute


def _make_estimator(model_type="linear", model_kwargs=None):
    """
    Factory for prediction models used in fitness evaluation.

    Parameters
    ----------
    model_type : str
        Type of model to fit. Supported:
            - "linear"     : sklearn.linear_model.LinearRegression
            - "ridge"      : sklearn.linear_model.Ridge
            - "lasso"      : sklearn.linear_model.Lasso
            - "elasticnet" : sklearn.linear_model.ElasticNet
            - "rf"         : sklearn.ensemble.RandomForestRegressor
    model_kwargs : dict or None
        Extra keyword arguments passed to the chosen estimator.

    Returns
    -------
    estimator : sklearn-like estimator with fit/predict methods
    """
    if model_kwargs is None:
        model_kwargs = {}

    if model_type == "linear":
        return LinearRegression(**model_kwargs)
    elif model_type == "ridge":
        return Ridge(**model_kwargs)
    elif model_type == "lasso":
        return Lasso(**model_kwargs)
    elif model_type == "elasticnet":
        return ElasticNet(**model_kwargs)
    elif model_type == "rf":
        # Reasonable defaults; user can override via model_kwargs
        default_kwargs = {"n_estimators": 100}
        # model_kwargs overrides defaults where specified
        default_kwargs.update(model_kwargs)
        return RandomForestRegressor(**default_kwargs)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         "Supported: 'linear', 'ridge', 'lasso', 'elasticnet', 'rf'.")


def _chromosome_fitness(chrom, X, y, folds, lambda_penalty,
                        model_type="linear", model_kwargs=None):
    """Compute penalized CV R^2 fitness for a single chromosome."""

    selected = np.where(chrom == 1)[0]
    if len(selected) == 0:
        return -np.inf

    X_sub = X[:, selected]
    r2_scores = []

    # Build model once per chromosome; refit inside each fold
    for train_idx, test_idx in folds:
        X_train, X_test = X_sub[train_idx], X_sub[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = _make_estimator(model_type=model_type, model_kwargs=model_kwargs)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2_scores.append(r2_score(y_test, preds))

    r2 = np.mean(r2_scores)
    penalty = lambda_penalty * (len(selected) / X.shape[1])
    return r2 - penalty


def evaluate_population_fitness(population, X, y, cv, lambda_penalty, rng,
                                model_type="linear", model_kwargs=None):
    """
    Compute penalized CV R^2 for the whole population.
    Parallelized over chromosomes with Dask.
    Converts chromosome --> fitness.

    Parameters
    ----------
    population : array (n_pop, n_features)
    X : array (n_samples, n_features)
    y : array (n_samples,)
    cv : int
        Number of CV folds.
    lambda_penalty : float
        Penalty weight λ; penalized fitness is R^2 - λ f where
        f is the fraction of selected predictors.
    rng : np.random.RandomState
        Not used directly here but kept for API compatibility.
    model_type : str, default="linear"
        See _make_estimator docstring.
    model_kwargs : dict or None
        Extra keyword arguments passed to the estimator.

    Returns
    -------
    fitness : np.ndarray (n_pop,)
        Penalized CV fitness values.
    """

    n_pop = population.shape[0]

    kf = KFold(n_splits=cv)
    folds = list(kf.split(X))

    tasks = [
        delayed(_chromosome_fitness)(
            chrom, X, y, folds, lambda_penalty,
            model_type=model_type,
            model_kwargs=model_kwargs
        )
        for chrom in population
    ]

    results = compute(*tasks)
    fitness = np.array(results, dtype=float)

    return fitness
