import numpy as np
import pandas as pd
import logging

from ._exceptions import GeneticAlgorithmException


def init_generate_population(search_space, population=10):
    """Generate an initial population for the genetic algorithm"""
    sample_set = [] # List of all samples taken

    for _ in range(population):

        sample = {}

        # Randomly generate set of parameters and add to dictionary
        for key, param in search_space.items():
            sample[key] = round(np.random.choice(param), 10)

        # Append random set of sample to sample_set
        sample_set.append(sample)
    
    return sample_set


def generate_offspring(genome_a, genome_b):
    """Generate two offspring using random crossover method from two sets of unique genomes"""
    # Validate that the genomes are of the same length
    if len(genome_a) != len(genome_b):
        logging.error(f"genome_a: {genome_a}")
        logging.error(f"genome_b: {genome_b}")
        raise GeneticAlgorithmException("Genome lengths do not match. Review genome pair passed to function")

    # Generate a random cross rate value used to split genomes
    random_cross_rate = np.random.choice(np.arange(1, len(genome_a), 1))

    # Split parents into two sides at position `random_cross_rate`
    keys, values = zip(*genome_a.items())
    split1_a = dict(zip(keys[:random_cross_rate], values[:random_cross_rate]))
    split2_a = dict(zip(keys[random_cross_rate:], values[random_cross_rate:]))

    keys, values = zip(*genome_b.items())
    split1_b = dict(zip(keys[:random_cross_rate], values[:random_cross_rate]))
    split2_b = dict(zip(keys[random_cross_rate:], values[random_cross_rate:]))

    # Generate offspring as crossed combination of parents
    offspring_a = {**split1_a, **split2_b}
    offspring_b = {**split1_b, **split2_a}

    return offspring_a, offspring_b


def roulette_wheel_selection(
    fitness_results:pd.DataFrame, param_labels:list, population:int=None, squared_prob=False,
):
    """Select set of genomes to reproduce using roulette wheel method for selection

    Parameters
    ----------
        fitness_results : pd.DataFrame
            DataFrame containing a fitness score column and any supporting
            statisticals desired with multiIndex of parameters. The fitness
            score must have the column name "fitness". 
        param_labels : list
            List of parameter labels add as key to parameter values.
        population : int or None, optional
            Returned population of genomes. If `population=None` [default],
            the returned population will be of the same length as the number
            of rows in the `fitness_results` DataFrame.
        squared_prob : bool, optional
            If `squared_prob=True` use squared fitness values to calculate 
            probabilities. Defaults to `False`.

    Returns
    -------
    list
        List of dictionaries containing genome parameter values keyed to parameter names. 
        Example:

        [
            {
                "param_1": 1.03, 
                "param_2": 2.34,
                "param_3": 3.41,
            },
        ]
    """
    # If a population is not specified...
    if not population:
        population = fitness_results.shape[0] # Set length to input row length

    # Convert fitness scores to probabilities
    # Log the probabilities as a fitness results column labeled "p"
    fitness_results["p"] = fitness_results["fitness"] / fitness_results["fitness"].sum()

    if squared_prob:
        # Restate the probabilities with squared fitness results
        fitness_results["p"] = fitness_results["fitness"] ** 2 / (fitness_results["fitness"] ** 2).sum()

    new_generation = fitness_results.sample(
        n=population,
        weights=fitness_results["p"],
        replace=True,
    )

    # Label parameter tupes with param names and wrap in dictionary
    # Append dictionary genomes to a list of all genomes in population
    genomes = []
    for gen in list(new_generation.index):
        g = dict(zip(param_labels, gen))
        genomes.append(g)

    return genomes


def crossover(parents:list, cross_rate:float=1.00):
    """Take in a list of parent genomes and produce random offspring
    
    Parameters
    ----------
        parents : list
            List of genomes represented by dictionary objects. Dictionaries should 
            be keys to parameter names with a single value (not an array or list) 
            for the parameter. Example of an appropriately formatted generation:

            [
                {
                    "param_1": 1.03, 
                    "param_2": 2.34,
                    "param_3": 3.41,
                },
            ]
        cross_rate : float, optional
            Probability that a parent's genome mates with another parent parent
            genome. If `cross_rate=1.00` (default), then 100% of parent genomes 
            will mate.

    Returns
    -------
    list
        Returns list of genomes represented by dictionaries in the identical format to 
        the `parents` parameter passed.  
    """
    generation = [] # Empty list for storage of next gneration of genomes

    # For each parent in our parent pairs, randomly generate offspring
    for parent in parents:
        # Select a spouse for the initial parent
        parents_ex_parent = parents.copy() 
        parents_ex_parent.remove(parent) # Remove parent from potential mates
        spouse = np.random.choice(parents_ex_parent)
        random_number = np.random.uniform(0, 1) # Random number between 0 and 1
        # Compare the random number to the probability of crossing our parents
        # If the random number exceeds the cross rate create random offspring
        if random_number < cross_rate:
            offspring, _ = generate_offspring(parent, spouse)
            generation.append(offspring)
        # If random_number exceed cross_rate, we will maintain parents unchanged
        if random_number > cross_rate:
            generation.append(parent)

    # If a parent list has an odd length, then add an additional genome to the
    # generation
    if len(generation) != len(parents):
        diff = np.abs(len(parents) - len(generation))
        generation.append(
            np.random.choice(parents, size=diff, replace=True)
        )

    return generation


def mutation(generation:list, param_space:dict, mutation_rate:float=0.05):
    """Randomly mutate parameters in the generation of genomes passed
    
    Parameters
    ----------
        generation : list
            List of genomes represented by dictionary objects. Dictionaries should 
            be keys to parameter names with a single value (not an array or list) 
            for the parameter. Example of an appropriately formatted generation:

            [
                {
                    "param_1": 1.03, 
                    "param_2": 2.34,
                    "param_3": 3.41,
                },
            ]
        param_space : dict
            Dictionary of lists or np.arrays describing the space of all
            possible parameters. Dictionary should be keyed in the same 
            way as the generation parameter's genomes dictionaries. Example:

            [
                {
                    "param_1": np.array([100,200,300])
                    "param_2": np.array([100,200,300])
                    "param_3": np.array([100,200,300])
                }
            ]
        mutation_rate : float, otpional
            Probability that an individual gene in a genome is mutated
        
    Returns
    -------
    list
        Returns list of genomes represented by dictionaries in the identical format to 
        the `generation` parameter passed.  
    """
    # For each genome...
    for genome in generation: 
        # If a random value is lower than the mutation probability...
        if np.random.uniform(0, 1) < mutation_rate:
            random_parameter = np.random.choice(list(param_space.keys()))
            gene = np.random.choice(param_space[random_parameter])
            # Update the genome with the mutant gene
            genome[random_parameter] = round(gene, 10)
    # Repeat for all genomes and genes
    return generation