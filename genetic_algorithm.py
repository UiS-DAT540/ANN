#%%
import numpy as np
import matplotlib.pyplot as plt

class GA:
    def __init__(self, entities, dna, population_size=50):
        """ Initialize genetic algorithm with properties
        
        Arguments:
            entities {class} -- Entities class which provides a framework for providing 
                                new entites as well as cloning existing ones.
            dna {class} -- Provides instructions for crossover and mutation
        
        Keyword Arguments:
            population_size {int} -- Number of entities per generation. It should be
                                     an even number. (default: {50})
        """
        self.entities = entities
        self.dna = dna
        self.n = population_size
        # ensure even number of entities
        assert self.n % 2 == 0

        self.fitness_over_time = []
        self.best_entities_over_time= []

        self.population = self.new_population()
        self.calculate_fitness()
        self.generation = 1

    def new_population(self):
        """Creatates a list of neural networks initialized with random weights.
            This will serve as initial population for the genetic algorithm.
        
        Returns:
            1d numpy array -- list with initialized neural networks
        """
        return np.array([self.entities.get_new_entity() for _ in range(self.n)])

    def calculate_fitness(self):
        """Calculate fitness for every member of the population
        """
        np.vectorize(lambda x: x.run_simulation())(self.population)
        

    def get_population_fitness(self):
        """ Return calculated fitness
        
        Returns:
            np.array -- Array of scores for each entity
        """
        # return np.vectorize(lambda x: x.reward)(self.population)
        return np.array(list(map(lambda x: x.reward, self.population)))

    def get_best_n_anns(self, n=1):
        """ Return n entities from population that have the highest fitness (reward).
            The entities can be in any order.

            Argpartition -  The n-th element will be in its final sorted position 
                            and all smaller elements will be moved before it and all 
                            larger elements behind it.

            negate fitness - to get the highest values before n insted of lowest
        
        Keyword Arguments:
            n {int} -- number of entities to return (default: {1})
        
        Returns:
            np.array -- array with n best entities
        """
        return self.population[(-self.get_population_fitness()).argpartition(n-1)[:n]]

    def select_n_from_distribution(self, n):
        """ Select n entities based on calculated probability distribution.
        
        Arguments:
            n {int} -- number of entities to select
        
        Returns:
            np.array -- array of n entities
        """
        return np.random.choice(self.population, size=n, replace=False, 
                                   p=self.get_relative_fitness())
    
    def get_relative_fitness(self):
        """ Calculate and return relative fitness of all entities in the population.
            Relative fitness is a probability distribution based on the fitness 
            of each entity in the population. In additon it sums to 1.
        
        Returns:
            np.array - array with relative fitness for each entity
        """
        fitness = self.get_population_fitness()
        return fitness/np.sum(fitness)

    def new_generation(self):
        """ Moves genetic algorithm forward one step. This means that algorithm, first, 
            selects two parents randomly based on the probability distribution of fitnesses. 
            Next, it produces 2 offsprings based on the rules set in the self.dna class.
            This is repeated until number of offsprings is equal to the population size.
            Lastly, the population is entirely replaced by offsprings and the generation 
            counter is increased.
        """
        new_generation = np.empty_like(self.population)
        for i in range(round(self.n/2)):
            parent1, parent2 = self.select_n_from_distribution(2)
            score = (parent1.reward + parent2.reward)/2
            child1, child2 = parent1.clone(), parent2.clone()
            self.dna.combine(child1.get_genes(), child2.get_genes(), score)
            new_generation[i*2:(i+1)*2] = child1, child2
        self.generation += 1
        self.population = new_generation
        self.calculate_fitness()

    def run(self, max_gen=20, max_score=None, plot=False):
        """ Run the genetic algoritm. Produce new generations of offsprings until
            maximum number of generations is reached, or we found an entity with
            wanted max_score.
        
        Keyword Arguments:
            max_gen {int} -- Number of generations at which the algorithm stops (default: {20})
            max_score {int} -- If max_score is set and reached by any entity, the algorithm
                               stops (default: {None})
            plot {bool} -- If true plot the evolution of average and highest reward. 
                           (default: {False}) 
        """
        self.report_generation_fitness()
        for _ in range(1, max_gen):
            if max_score and self.get_best_n_anns(1)[0].reward >= max_score:
                break
            self.new_generation()
            self.report_generation_fitness()
        if plot:
            self.plot_timeline()

    def report_generation_fitness(self):
        """ Get average and best scores and store them in a list.
            Additionaly, print the information about the current 
            generation.
        """
        # calc and set information
        popultation_fitness = self.get_population_fitness()
        avg = popultation_fitness.mean()
        best = popultation_fitness.max()
        if self.generation > len(self.fitness_over_time):
            self.fitness_over_time.append([avg, best])
            self.best_entities_over_time.append(self.get_best_n_anns(1)[0])

        # print information
        print("Generation {}:".format(self.generation))
        print("\tAverage score:\t{}".format(avg))
        print("\tBest score: \t{}\n".format(best))

    def plot_timeline(self):
        """ Plot the evolution of scores of best entities and average of 
            all entities in the population.
        """
        plt.plot(np.arange(self.generation)+1, self.fitness_over_time)
        plt.title('Training')
        plt.legend(['average', 'best'])
        plt.ylabel("score")
        plt.xlabel("generation")
        plt.show()

    def render_best(self, max_iter=None):
        """ Find and render best entity
        
        Keyword Arguments:
            max_iter {int} -- If set, stop rendering after max_iter steps (default: {None})
        """
        best_entity = self.best_entities_over_time[-1]
        best_entity.run_simulation(render=True, max_iter=max_iter)
        print('Score for this run is: {}'.format(best_entity.reward))

    def render_average(self, max_iter=None):
        """ Calculate average coeficients from all entities in the last generation.
            Create new entity with calculated coefficients and render.
        
        Keyword Arguments:
            max_iter {int} -- If set, stop rendering after max_iter steps (default: {None})
        """
        coefs = list(map(lambda x: x.coefs_, self.population))
        avg_coef = [sum(map(lambda x: x[i], coefs))/self.n for i in range(len(coefs[0]))]

        intercepts = list(map(lambda x: x.intercepts_, self.population))
        avg_intercept = [sum(map(lambda x: x[i], intercepts))/self.n for i in range(len(intercepts[0]))]

        # %%
        avg_entity = self.entities.get_new_entity(coefs_=avg_coef, intercepts_=avg_intercept)
        avg_entity.run_simulation(render=True, max_iter=max_iter)


# %%
