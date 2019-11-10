#%%
import numpy as np

class DNA():
    """ Class contining methods for crossover and mutation

    
    Raises:
        Exception: invalid crossover method
    """
    def __init__(self, crossover_method="one-point", crossover_ravel=False, 
                mutation_method=None, mutation_rate=0.001, mutation_mag=1):
        self.crossover_method = crossover_method
        self.crossover_ravel = crossover_ravel
        self.mutation_rate = mutation_rate
        self.mutation_method = mutation_method
        self.mutation_mag = mutation_mag

    def combine(self, dna1, dna2, score=None):
        """ Main method of the class. Mixes and mutates two dna strings 
            (weights of a neural network). Dna1 and dna2 are lists of 
            numpy arrays that hold coefficients to be mixed and mutated.
            If mutation_rate is a function, it will be called with
            'score' as a parameter.
        
        Arguments:
            dna1 {list} -- list of numpy arrays
            dna2 {list} -- list of numpy arrays
        
        Keyword Arguments:
            score {float} -- score to be provided to mutation_rate function
                            (default: {None})
        
        Returns:
            tuple -- returns mutated and mixed dna1 and dna2
        """
        dna1, dna2 = self.crossover(dna1, dna2)
        mutation_rate = self.mutation_rate 
        if callable(mutation_rate): 
            mutation_rate = self.mutation_rate(score)
        self.mutate(dna1, mutation_rate)
        self.mutate(dna2, mutation_rate)
        return dna1, dna2

    def mutate(self, genes, mutation_rate):
        """ For every element (np.array) in a list, mutate those elements if
            needed.
        
        Arguments:
            genes {list} -- list of np.arrays
            mutation_rate {int} -- dictates frequency of mutation
        
        Returns:
            list -- same format as genes
        """
        return [self._mutate_row(row, mutation_rate) for row in genes]
        
    def _mutate_row(self, row, mutation_rate):
        """ Every element in row has a chance (given by mutation rate)
            to be mutated. Mutations can be different according to
            mutation_method:
                'nullify' - Set selected elements to 0

                'permutate' - Randomly shuffle selectedd elemnets.
                              Notice that this is only applied if there
                              are 2 or more elements that need to be mutated
                            
                'uniform' - Add uniform value [-1, 1] to each selected element.

                'normal' - Add value from a normal distribution ~N(0,1) to each
                            element.
        
        Arguments:
            row {np.array} -- np.array where each element can be mutated.
            mutation_rate {int} -- Chance for an element to be mutated
        
        Returns:
            np.array -- same format as row, but with mutated elements
        """
        # setup
        shape = row.shape
        row = row.ravel()

        # mutate
        mutation_mask = np.random.rand(len(row)) < mutation_rate
        if self.mutation_method == 'nullify':
            row[mutation_mask] = 0
        elif self.mutation_method == 'permutate':
            row[mutation_mask] = np.random.permutation(row[mutation_mask])
        elif self.mutation_method == 'uniform':
            row[mutation_mask] += np.random.uniform(-1.0, 1.0, sum(mutation_mask))*self.mutation_mag
        elif self.mutation_method == 'normal':
            row[mutation_mask] += np.random.normal(size=sum(mutation_mask))*self.mutation_mag
        
        #return
        return row.reshape(shape)

    def crossover(self, dna1, dna2):
        """ Call crossover function on every pair of numpy arrays in dna1 and dna2.
        
        Arguments:
            dna1 {list} -- list of numpy arrays
            dna2 {list} -- list of numpy arrays. It should have the same
                           structure as dna1
        
        Returns:
            tuple -- dna1, dna2 but with mixed values.
        """
        for row1, row2 in zip(dna1, dna2):
            row1, row2 = self._crossover_row(row1, row2)
        return dna1, dna2

    def _crossover_row(self, row1, row2):
        """ Replaces some of the values in row1 with the values in row2. Values that
            were replaced are put into row2 at the same indices. There are several method 
            we can use to choose which elements should be swapped:
                'one-point' - Find a single index and swap the values before that index.

                'two-point' - Choose two indices and swap all elements between those 
                              indices.
                            
                'uniform' - Every element has a 50% chance to be swapped.

            In addition we can use this three method on rows that are raveled (made into 1D 
            array) or keep the original multidimensional form.
        
        Arguments:
            row1 {np.array} -- numpy array holding values to be swapped with row2
            row2 {np.array} -- numpy array holding values to be swapped with row1.
                                It should have the same shape as row1
        
        Raises:
            Exception: Invalid crossover_method
        
        Returns:
            tuple -- row1 and row2 with shuffled values
        """
        # setup
        if self.crossover_ravel:
            shape = row1.shape
            row1, row2 = row1.ravel(), row2.ravel()
            length = len(row1)
        else:
            length = row1.shape[-1]

        # type of crossover
        if self.crossover_method == "one-point":
            mask = slice(np.random.randint(length+1))
        elif self.crossover_method == "two-point":
            mask = slice(*np.sort(np.random.randint(length+1, size=2)))
        elif self.crossover_method == "uniform":
            mask = np.random.rand(length) < 0.5
        else:
            raise Exception("Invalid crossover type.")

        # do crossover
        if len(row1.shape) == 1:
            tmp = row2[mask].copy()
            row2[mask], row1[mask]  = row1[mask], tmp
        else:
            tmp = row2[:,mask].copy()
            row2[:,mask], row1[:,mask]  = row1[:,mask], tmp

        # return 
        if self.crossover_ravel:
            return row1.reshape(shape), row2.reshape(shape)
        return row1, row2

# %%
