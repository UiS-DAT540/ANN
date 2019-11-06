#%%
import gym
import numpy as np
from neural_network import ANN
from genetic_algorithm import GA

#%%
# setup
n = 50 # number of classifiers per generation
num_gen = 50 # number of generations
hlayer_size = (4,)
crossover_method = "uniform"
# mutation rate [0.1 - 0.001]
# mutation_rate = lambda x: 0.1/np.log(x)
mutation_rate = lambda x: 5/x

# init enviroment
env = gym.make('CartPole-v1')
env._max_episode_steps = np.inf

# train
ga = GA(env, ANN, population_size=50, crossover_method=crossover_method, 
        partial_fit=False, max_repetitions=None, mutation_rate=mutation_rate)
ga.run(max_gen=num_gen)
# ga.render_best()

# close environment
env.close()

# %%
