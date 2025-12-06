import numpy as np 

def select_parents(population, fitness, rng):
    """We are doing rank based seelection: who becomes a parent?"""

    ranks = np.argsort((np.argsort(fitness)))   #Convert fitness to rank 0 ... n_pop-1
    probs = (ranks + 1) / np.sum(ranks + 1)     # worst rnak gets weight=1 , best rnak gets weigth=n_pop and normalize
    
    # Randomly sample parent.  god fitness --> highe rprob 
    idx = rng.choice(len(population), size=len(population), p=probs)    

    # return array of parent chromosomes 
    return population[idx]
