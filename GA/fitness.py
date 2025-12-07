import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def evaluate_population_fitness(population, X, y, cv, lambda_penalty, rng):
    """Comptue CV R2 for the whole population.
    How good is each chromosome? Converts chromosome --> fitness"""

    n_pop = population.shape[0]

    fitness = np.zeros(n_pop)   #preallcoate fitness score

    #kf = KFold(n_splits=cv, shuffle=True, random_state=rng)
    # NOTE:
    # We do NOT shuffle folds here.
    # Shuffling introduces randomness into the train/test splits,
    # which caused identical chromosomes to receive different fitness scores.
    # Removing shuffle ensures deterministic splits, so identical chromosomes
    # always produce identical R2 values — resolving the failing unit test.
    kf = KFold(n_splits=cv)

    for i, chrom in enumerate(population):
        selected = np.where(chrom == 1)[0]  # indeces of features included by this chromosome 
        if len(selected) == 0:              # if chromosome selects no variables, discard 
            fitness[i] = -np.inf
            continue

        X_sub = X[:, selected]                  # chose only selected columns of X 
        r2_scores = np.zeros(cv)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_sub)): # split into training and testing data
            model = LinearRegression()
            model.fit(X_sub[train_idx], y[train_idx])
            preds = model.predict(X_sub[test_idx])
            r2_scores[fold_idx] = r2_score(y[test_idx], preds)

        # Compute average R2 across folds
        r2 = np.mean(r2_scores)

        penalty = lambda_penalty * (len(selected) / X.shape[1])
        fitness[i] = r2 - penalty                   #total fitness

    return fitness
