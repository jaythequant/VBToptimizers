import pandas as pd
import numpy as np
import warnings
from itertools import chain
from .operators import mutation
from ._exceptions import GeneticAlgorithmException


def _make_numpy_dictionary(params:dict):
    """Convert params from `generate_population` to dictionary of numpy arrays"""
    output = {}

    df = pd.DataFrame(params) # Wrap params in dataframe
    df_list = df.to_dict("list") # Convert df to wll formatted dict

    for key, values in df_list.items():
        output[key] = values # Convert lists in df_list to np.array

    return output


def _restate_constants(const_dict:dict, iterations) -> list:
    """Convert constants dictionary to tuples"""
    lst = []

    for k, v in const_dict.items():
        s = iterations * k
        lst.append((s,v))

    return lst


def _handle_duplication(generation:list, param_space:dict, handler:str="random", step_size:float=0.10):
    """Manage the handling of duplicate genomes from generation to generation.
    
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
        handler : str, otpional
            Handler parameter controls how `_handle_duplication` will correct duplicate
            genomes within a generation. The handler excepts "mutate" or `None`. If `mutate`,
            force mutations in duplicate genomes till generation only contains unique genome
            sets. If `None`, allow generations to contains genomes. Note that this may cause 
            early convergence.
        
    Returns
    -------
    list
        Returns list of genomes represented by dictionaries in the identical format to 
        the `generation` parameter passed.  
    """
    df = pd.DataFrame(generation)
    iter_count = 0 # Max iterations before breaking while loop
    while df.duplicated().any():
        iter_count += 1
        if iter_count < 100:
            # Isolate the duplicates in their own dictionary
            duplicates = df[df.duplicated()].to_dict("records")
            # Mutate the dictionaries to be unique values
            duplicates = pd.DataFrame(
                mutation(
                    duplicates, param_space, mutation_rate=1.00, 
                    style=handler, step_size=step_size,
                )
            )
            df = df.drop_duplicates(keep="first")
            df = pd.concat([df, duplicates]).reset_index(drop=True)
        elif iter_count > 100:
            warnings.warn("Returned dataframe contains duplicates")
            break
    return df


def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _batch_populations(population:list, n_batch_size:int=None, n_batches:int=None):
    """Split population into managable batches"""
    batches = []

    if n_batches and n_batch_size:
        raise GeneticAlgorithmException("Cannot specify both n_batch_size and n_batches")

    # Do not batch
    if not n_batch_size and not n_batches:
        return _make_numpy_dictionary(population)

    # Split into unspecified batches of size n
    if n_batch_size and not n_batches:
        chunk_gen = _chunks(population, n_batch_size)
        for chunk in chunk_gen:
            batch = _make_numpy_dictionary(chunk)
            batches.append(batch)

    # Split into n batches with unspecified size
    # Currently this part does not work
    if n_batches and not n_batch_size:
        np_batches = np.array_split(population, n_batches)
        for b in np_batches:
            # batch = _make_numpy_dictionary(list(b))
            batches.append(b) # Convert np.array to list
        batches = list(chain.from_iterable(batches))

    return batches
