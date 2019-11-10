#%%
import itertools, os
import numpy as np
import pandas as pd
from run import run

# This is not relevant for the project, but it can be included.
# Set of functions meant to run the genetic algorithm for all 
# specified configurations, store the results in a dataframe 
# and save to file. 
# 

def get_dataframe_with_params():
    """ Get all combinations of selected configurations and store 
        it in a dataframe

    Returns:
        pd.DataFrame -- Dataframe with all config combinations
    """
    # values to use
    hlayer_size = [(1,), (4,)]
    activation = ['logistic', 'tanh', 'relu', 'identity']
    crossover_method = ['one-point', 'two-point', 'uniform']
    mutation_method = ['nullify', 'permutate', 'normal', 'uniform']
    mutation_rate = [0.001, 0.01, 0.1, 0.5, 'lambda s: 5/s']
    param_collection = {'hlayer_size':hlayer_size, 'activation':activation, 
        'crossover_method':crossover_method, 'mutation_method':mutation_method,
        'mutation_rate':mutation_rate}
    # get combinations
    combinations = itertools.product(*param_collection.values())
    # init dataframe with all combinations
    df = pd.DataFrame(combinations, columns=param_collection.keys())
    return df

def save_df_to_file(df, filename="data\\results.csv"):
    """ Helper to save the resulting dataframe to file
    
    Arguments:
        df {pd.DataFrame} -- DataFrame to be saved to file
    
    Keyword Arguments:
        filename {str} -- Name of the file where to save the dataframe
                            (default: {"data\results.csv"})
    """ 
    path = os.path.join(os.getcwd(), filename)
    df.to_csv(path)

def read_df_from_file(filename="data\\results.csv"):
    """ Helper to read dataframe from file
    
    Keyword Arguments:
        filename {str} -- filename to open (default: {"data\results.csv"})
    
    Returns:
        pd.DataFrame -- Dataframe that holds the file data
    """
    return pd.read_csv(filename)

def get_results_for_all_combinations(df):
    """ Loads dataframe with all combinations of configurations.
        Runs the genetic algorithm for all combinations and adds the
        results to the dataframe. Results are best score, average 
        score, generation number for those scores.

    Returns:
        pd.DataFrame -- Dataframe with all combinations and results of those 
                        combinations. 
    """ 
    res = df.apply(get_results_for_single_config, axis=1)
    gen, avg, best = np.concatenate(res.values).T
    return df.reindex(df.index.repeat(res.str.len())).assign(gen=gen, avg=avg, best=best)

def get_results_for_single_config(series):
    """ Run the genetic algorithm with a single configuration of
        parameters.

    Arguments:
        series {pd.Series} -- pd.Series with parameter values
    
    Returns:
        list -- list of results for generation number, average and best scores
    """
    print('Runing genetic algorithm for values:')
    print(series, '\n')
    ga = run('CartPole-v1', *series.values, max_score=1000)
    return [[i+1,score[0],score[1]] for i,score in enumerate(ga.fitness_over_time)]



if __name__ == "__main__":
    # IT TAKES SOME TIME TO FINISH
    df = get_dataframe_with_params()
    # df = get_results_for_all_combinations(df)
    # save_df_to_file(df)

# %%
