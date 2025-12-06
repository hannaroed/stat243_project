import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def evaluate_population_fitness(population, X, y, cv, lambda_penalty, rng):
    """Comptue CV R2 for the whole population.
    How good is each chromosome? Converts chromosome --> fitness"""

    n_pop = population.shape[0]

    fitness = np.zeros(n_pop)   #preallcoate fitness score

    kf = KFold(n_splits=cv, shuffle=True, random_state=rng)

    for i, chrom in enumerate(population):
        selected = np.where(chrom == 1)[0]  # indeces of features included by this chromosome 
        if len(selected) == 0:              # if chromosome selects no variables, discard 
            fitness[i] = -np.inf
            continue

        X_sub = X[:, selected]                  # chose only selected columns of X 
        preds = np.zeros_like(y, dtype=float)

        for train_idx, test_idx in kf.split(X_sub): # split into training and testing data
            model = LinearRegression()
            model.fit(X_sub[train_idx], y[train_idx])
            preds[test_idx] = model.predict(X_sub[test_idx])

        # Compute R2 loss 
        ss_res = np.sum((y - preds) ** 2)           #sum of squared prediction errors
        ss_tot = np.sum((y - np.mean(y)) ** 2)      #total variability around the mean
        r2 = 1 - ss_res / ss_tot                    #How much variability did this model explain?

        penalty = lambda_penalty * (len(selected) / X.shape[1])
        fitness[i] = r2 - penalty                   #totla fitness

    return fitness




        
