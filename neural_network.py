#%%
import numpy as np
from sklearn.neural_network import MLPClassifier
import copy


class ANN(MLPClassifier):
    def __init__(self, env, hidden_layer_sizes=None, **kwargs):
        super().__init__(batch_size=1, max_iter=1, solver='sgd', 
                        activation='tanh', learning_rate='invscaling', 
                        random_state=1)

        # number of neurons in the hidden layer, if not specified:
        # 1 layer with 2/3 input layer + 1 for output layer = 4
        self.hidden_layer_sizes = hidden_layer_sizes \
            if hidden_layer_sizes \
            else (round((env.observation_space.shape[0] + 1) * 2 / 3) + 1, )
        
        self.partial_fit(
            np.array([env.observation_space.sample()]),
            # np.array([env.reset()]),
            np.array([env.action_space.sample()]), 
            classes=np.arange(env.action_space.n))
        
        # assign variables
        self.env = env
        self.__dict__.update(kwargs)

    def run_simulation(self, partial_fit=False, max_repetition=None, max_iter=1000, render=False):
        next_obs = self.env.reset()
        next_action = None

        if max_repetition:
            current_repetition = 0

        for t in range(max_iter):
            obs = next_obs
            if render:
                self.env.render()
            prev_action, next_action = next_action, self.predict(obs.reshape(1, -1))[0]
            if max_repetition:
                current_repetition = current_repetition + 1 if next_action == prev_action else 0
                if current_repetition >= max_repetition:
                    next_action = self.env.action_space.sample()
                    current_repetition = current_repetition if next_action == prev_action else 0

            next_obs, _, done, _ = self.env.step(next_action)
            if done:
                break
            if partial_fit:
                self.partial_fit(
                    np.array([obs]),
                    np.array([next_action])
                )
        self.reward = t

    def mutate(self, *args):
        for row in self.coefs_+self.intercepts_:
            row = self._mutate_arr(row, *args)
            
    def _mutate_arr(self, arr, mutation_rate=0.1, mag=100):
        shape = arr.shape
        arr = arr.ravel()

        mutation_mask = np.random.rand(len(arr)) < mutation_rate
        # arr[mutation_mask] = 0
        # arr[mutation_mask] = np.random.permutation(arr[mutation_mask])
        # arr[mutation_mask] += np.random.uniform(-1.0, 1.0, len(arr[mutation_mask]))
        arr[mutation_mask] = np.random.normal(size=len(arr[mutation_mask]))
        # arr[mutation_mask] += np.random.normal(size=len(arr[mutation_mask]))*mag
        return arr.reshape(shape)

    def crossover(self, other, *args):
        for row1, row2 in zip(self.coefs_+self.intercepts_, other.coefs_+other.intercepts_):
            row1, row2 = self._crossover_rows(row1, row2, *args)

    def _crossover_rows(self, row1, row2, crossover_method="test", ravel=False):
        # setup
        if ravel:
            shape = row1.shape
            row1, row2 = row1.ravel(), row2.ravel()
            length = len(row1)
        else:
            length = row1.shape[-1]

        # type of crossover
        if crossover_method == "one-point":
            mask = slice(np.random.randint(length+1))
        elif crossover_method == "two-point":
            mask = slice(*np.sort(np.random.randint(length+1, size=2)))
        elif crossover_method == "uniform":
            mask = np.random.rand(length) < 0.5
        else:
            raise Exception("Invvalid crossover type.")

        # do crossover
        if len(row1.shape) == 1:
            tmp = row2[mask].copy()
            row2[mask], row1[mask]  = row1[mask], tmp
        else:
            tmp = row2[:,mask].copy()
            row2[:,mask], row1[:,mask]  = row1[:,mask], tmp

        # return 
        if ravel:
            return row1.reshape(shape), row2.reshape(shape)
        return row1, row2

    def clone(self):
        return ANN(self.env, self.hidden_layer_sizes, 
            coefs_=copy.deepcopy(self.coefs_), intercepts_=copy.deepcopy(self.intercepts_))



# %%
