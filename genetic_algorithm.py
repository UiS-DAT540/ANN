#%%
import numpy as np
import matplotlib.pyplot as plt

class GA:
    def __init__(self, env, ann, population_size=50, crossover_method="uniform", 
                partial_fit=False, max_repetitions=None, max_iter=10000, mutation_rate=0.001):
        self.env = env
        self.ANN = ann
        self.n = population_size

        self.max_iter = max_iter
        self.partial_fit = partial_fit
        self.max_repetitions =  max_repetitions
        self.crossover_method = crossover_method
        self.mutation_rate = mutation_rate
        self.fitness_all = []
        self.best_agents = []
        self.population = self.new_population()
        self.train_population()
        self.generation = 1

    def new_population(self):
        return np.array([self.ANN(self.env) for i in range(self.n)])

    def train_population(self):
        np.vectorize(lambda x: x.run_simulation(max_iter=self.max_iter, partial_fit=self.partial_fit,
                            max_repetition=self.max_repetitions))(self.population)
        

    def get_population_fitness(self):
        return np.vectorize(lambda x: x.reward)(self.population)

    def get_best_n_anns(self, n=1):
        return self.population[(-self.get_population_fitness()).argpartition(n-1)[:n]]

    def select_n_from_distribution(self, n):
        return np.random.choice(self.population, size=n, replace=False, 
                                   p=self.get_relative_fitness())
    
    def get_relative_fitness(self):
        fitness = self.get_population_fitness()
        return fitness/np.sum(fitness)

    def new_generation(self):
        new_generation = np.empty_like(self.population)
        for i in range(round(self.n/2)):
            new_generation[i*2:(i+1)*2] = self.breed(*self.select_n_from_distribution(2))
        self.generation += 1
        self.population = new_generation
        self.train_population()
            
    def breed(self, parent1, parent2):
        child1, child2 = parent1.clone(), parent2.clone()
        child1.crossover(child2, self.crossover_method, False)
        avg_reward = (parent1.reward + parent2.reward)/2
        mutation_rate = self.mutation_rate if not callable(self.mutation_rate) else self.mutation_rate(avg_reward)
        child1.mutate(mutation_rate)
        child2.mutate(mutation_rate)
        return child1, child2

    def run(self, max_gen):
        self.report_generation_fitness()
        for _ in range(max_gen-1):
            self.new_generation()
            self.report_generation_fitness()
            # if self.get_best_n_anns(1)[0].reward == self.max_iter-1:
            #     break
        self.plot_timeline()

    def report_generation_fitness(self):
        popultation_fitness = self.get_population_fitness()
        avg = popultation_fitness.mean()
        best = popultation_fitness.max()
        if self.generation > len(self.fitness_all):
            self.fitness_all.append([avg, best])
            self.best_agents.append(self.get_best_n_anns(1)[0])
        print("Generation {}:".format(self.generation))
        print("\tAverage reward:\t{}".format(avg))
        print("\tBest reward: \t{}\n".format(best))

    def plot_timeline(self):
        plt.plot(self.fitness_all)
        plt.title('Training')
        plt.legend(['average', 'best'])
        plt.ylabel("score")
        plt.xlabel("generation")

    def render_best(self):
        self.best_agents[-1].run_simulation(render=True, max_iter=1000)

#%%