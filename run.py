#%%
import gym
import numpy as np
from neural_network import Entities
from genetic_algorithm import GA
from dna import DNA

# Function to run the genetic algorithm for a single configuration.
# This is the main script to run.

#%%
def run(env_name, hlayer_size, activation, crossover_method, mutation_method,
        mutation_rate, population_size=50, num_gen=10, plot=False,
        max_score=10000):
    """ Runs the genetic algorithm with defined properties. 
        Returns the genetic algorithm class after num_gen have run.
    
    Arguments:
        env_name {str} -- environmnet to load from gym library
        hlayer_size {tuple} -- neural network shape
        activation {str} -- neural network activation function
        crossover_method {str} -- method to use when doing crossover
        mutation_method {str} -- method to use when doing mutation
        mutation_rate {float|function} -- frequency of mutation. If
                        function, it is called with average score 
                        of parents as a parameter.
    
    Keyword Arguments:
        population_size {int} -- number of classifier in each generation
                                 (default: {50})
        num_gen {int} -- Number of generations the algorithm runs over.
                        It might be ignored if max_score is reached
                         (default: {10})
        plot {bool} -- Whether to plot the timeline progression of best
                        and average scores over generations (default: {False})
        max_score {int} -- If any entity in the population reaches 
                            the set score, algorithm stops. (default: {10000})
    
    Returns:
        class -- The genetic algorithm with population after 
                number of generations.
    """

    if type(mutation_rate) == str:
        mutation_rate = eval(mutation_rate)

    # init enviroment
    env = gym.make(env_name)
    env._max_episode_steps = np.inf

    # init entities
    ent = Entities(env=env, hidden_layer_sizes=hlayer_size, 
                activation=activation, do_partial_fit=False, 
                max_repetition=None, max_iter=10000)

    # init dna
    dna = DNA(crossover_method=crossover_method, crossover_ravel=False, 
            mutation_method=mutation_method, mutation_rate=mutation_rate, mutation_mag=1)

    # initialize the genetic algorithm
    ga = GA(ent, dna, population_size=population_size)
    # run the genetic algorithm
    ga.run(max_gen=num_gen, max_score=max_score, plot=plot)
    # ga.render_best(max_iter=1000)
    # ga.render_average(max_iter=1000)
    # close
    env.close()
    return ga

# %%
if __name__ == "__main__":
    # genetic algorithm options
    # number of classifiers per generation
    population_size = 50 
    # max number of generations, 
    # usualy doesn't take this long to stop
    num_gen = 10
    # environment name
    env_name = 'CartPole-v1'

    # neural network options
    hlayer_size = (1,) # [(1,), (4,)]
    activation = 'tanh' # ['identity', 'logistic', 'tanh', 'relu']

    # dna options
    crossover_method = 'one-point' # ['one-point', 'two-point', 'uniform']
    mutation_method = 'normal' # ['nullify', 'permutate', 'normal', 'uniform']
    mutation_rate = lambda s: 5/s # [0.001, 0.01, 0.1, 0.5, lambda s: 5/s]
    
    print('Configuration: ')
    print('Environment name: \t', env_name)
    print('Population size: \t', population_size)
    print('Number of generations: \t', num_gen)
    print('Hidden layer shape: \t', hlayer_size)
    print('Activation function: \t', activation)
    print('Crossover method: \t', crossover_method)
    print('Mutation method: \t', mutation_method)
    print('Mutation rate: \t\t', mutation_rate)
    print()
    
    # run
    ga = run(env_name, hlayer_size, activation, crossover_method, mutation_method, 
        mutation_rate, population_size, num_gen, True, 10000)

# %%