import numpy as np
from tqdm import tqdm
from .population import initialize_population
from .fitness import evaluate_population_fitness
from .selection import select_parents
from .crossover import crossover_population
from .mutation import mutate_population


def select(X, y, n_pop=50, n_gen=50, mutation_rate=0.01,
           crossover_rate=0.8, lambda_penalty=0.0, cv=5,
           random_state=None, feature_names=None):
    """
    Driver function for GA Algorithm 
    
    Parameters
    -----------
    X: array (n_samples, n features)
    y: array (n_samples,)

    n_pop: int (Populaiton size / number of chromsooems )

    n_gen: int (Number of generations to evolve)

    mutation_rate: float (Probability gene flips)

    crossover_rate: float (Probability crossover occurs for a parent pair)

    cv: int ( number of cross valdiaiotn folds)

    Returns
    ---------

    dictionary: 
        best chromosome: np.ndarray (0/1 variable inclusion)
        selected_vars : list of selected variable indices
        best_fitness : float (CV R^2)
        fitness_history : list of best fitness per generation

    """

    rng = np.random.RandomState(random_state)

    n_features = X.shape[1]
    population = initialize_population(n_pop, n_features, rng)

    fitness_history = []

    for gen in tqdm(range(n_gen), desc="GA generations", unit="gen"):

        # Evaluate Fitness ------------------------------------------------------
        fitness = evaluate_population_fitness(population, X, y, cv, lambda_penalty, rng)

        # Trakc best results
        best_idx = np.argmax(fitness) 
        fitness_history.append(fitness[best_idx]) # track fitness

        # Parent Selection ------------------------------------------------------

        parents = select_parents( population, fitness, rng)

        # Crossover ------------------------------------------------------

        offspring = crossover_population( parents, crossover_rate, rng)

        # Mutation ------------------------------------------------------

        mutated = mutate_population(offspring, mutation_rate, rng)

        # Define new population 

        population = mutated.copy()
    
    # Final evaluation

    fitness = evaluate_population_fitness(population, X, y, cv, lambda_penalty, rng)

    best_idx = np.argmax(fitness) 
    best_chromosome = population[best_idx]

    selected_vars = [int(i) for i in np.where(best_chromosome == 1)[0]]
    selected_var_names = None

    if feature_names is not None:
        selected_var_names = [feature_names[i] for i in selected_vars]

    return {
        "best_chromosome": best_chromosome,
        "selected_vars": selected_vars,
        "selected_var_names": selected_var_names,
        "n_selected": len(selected_vars),
        "best_fitness": fitness[best_idx],
        "fitness_history": fitness_history
    }



    # selected_vars = list(np.where(best_chromosome == 1)[0])

    
    # return {
    #     "best_chromosome": best_chromosome,
    #     "selected_vars": selected_vars,
    #     "best_fitness": fitness[best_idx],
    #     "fitness_history": fitness_history
    # }