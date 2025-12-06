import numpy as np
from project.select import select


def toy_regression_data(seed=0, n_samples=60, n_features=6):
    """
    Small synthetic regression dataset for testing the GA driver.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = 1.5 * X[:, 0] - 0.7 * X[:, 2] + 0.1 * rng.randn(n_samples)
    return X, y


def test_select_output_structure():
    """
    select() should return a dictionary with required keys
    and internally consistent values.
    """
    X, y = toy_regression_data()

    params = {
        "n_pop": 20,
        "n_gen": 5,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "cv": 3,
        "lambda_penalty": 0.0,
        "random_state": 42,
    }

    result = select(X, y, params)

    assert isinstance(result, dict)
    for key in ("best_chromosome", "selected_vars",
                "best_fitness", "fitness_history"):
        assert key in result

    best_chr = result["best_chromosome"]
    selected_vars = result["selected_vars"]

    assert set(np.unique(best_chr)).issubset({0, 1})
    assert selected_vars == list(np.where(best_chr == 1)[0])
    assert len(result["fitness_history"]) == params["n_gen"]


def test_select_reproducibility_with_seed():
    """
    Using the same random_state should produce identical results.
    """
    X, y = toy_regression_data()

    params = {
        "n_pop": 15,
        "n_gen": 4,
        "mutation_rate": 0.2,
        "crossover_rate": 0.7,
        "cv": 3,
        "lambda_penalty": 0.0,
        "random_state": 123,
    }

    res1 = select(X, y, params)
    res2 = select(X, y, params)

    assert np.array_equal(res1["best_chromosome"], res2["best_chromosome"])
    assert res1["best_fitness"] == res2["best_fitness"]
    assert res1["fitness_history"] == res2["fitness_history"]