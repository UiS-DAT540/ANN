#%%
import numpy as np
from sklearn.neural_network import MLPClassifier
import copy

class Entities:
    """ Helper class to store parameters that all our neural networks share
    """
    
    def __init__(self, env, hidden_layer_sizes=None, solver='sgd', 
                    activation='relu', learning_rate='invscaling', 
                    random_state=1, do_partial_fit=False, max_repetition=None,
                    max_iter=10000):

        self.env = env
        self.hidden_layer_sizes = hidden_layer_sizes
        self.solver = solver
        self.activation = activation
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.do_partial_fit = do_partial_fit
        self.max_repetition = max_repetition
        self.max_iter = max_iter

    def get_new_entity(self, **kwargs):
        """ Return new neural network with initialized weights
        
        Returns:
            class -- ANN class, basicaly initialized MLPClassifier 
        """
        return ANN(self, **kwargs)
        
class ANN(MLPClassifier):
    """ Inherits from MLPClassifier. This class initializes the
        neural network and has some extra methods like clone(),
        get_genes() and run_simulation().
    
    Arguments:
        MLPClassifier {class} -- Neural network
    
    Returns:
        class -- Neural network
    """
    def __init__(self, framework, **kwargs):
        super().__init__(batch_size=1, max_iter=1, solver=framework.solver, 
                        activation=framework.activation, learning_rate=framework.learning_rate, 
                        random_state=framework.random_state)

        self.env = framework.env
        self.hidden_layer_sizes = framework.hidden_layer_sizes
        self.solver = framework.solver
        self.activation = framework.activation
        self.learning_rate = framework.learning_rate
        self.random_state = framework.random_state
        self.do_partial_fit = framework.do_partial_fit
        self.max_repetition = framework.max_repetition
        self.max_iter = framework.max_iter

        # number of neurons in the hidden layer, if not specified:
        # 1 layer with 2/3 input layer + 1 for output layer = 4
        if not self.hidden_layer_sizes:
            self.hidden_layer_sizes = (round(self.env.observation_space.shape[0] * 2/3) + 1, )
        
        self.partial_fit(
            np.array([self.env.observation_space.sample()]),
            # np.array([env.reset()]),
            np.array([self.env.action_space.sample()]), 
            classes=np.arange(self.env.action_space.n))
        
        # other variables
        self.framework = framework
        self.__dict__.update(kwargs)

    def get_genes(self):
        """ Returns list of coeficients and biases. They are concatenated for simplicity.
        
        Returns:
            list -- list of numpy arrays with weight for this neural network
        """
        return self.coefs_ + self.intercepts_

    def clone(self):
        """ return clone of this class. Weight are lists holding numpy arrays. In order to 
            avoid seting views into weight to the new classifier we have to clone it deep. 

        
        Returns:
            class -- return ANN with same weights as this class
        """
        return ANN(self.framework, coefs_=copy.deepcopy(self.coefs_), 
                    intercepts_=copy.deepcopy(self.intercepts_))

    def run_simulation(self, render=False, max_iter=None):
        """ Run the simulation with this classifier in the given environment.
            The simulation ends when eiter max_iter is reached, or the classifier
            choose the wrong action, resulting in no reward.

            If max_repetition is set, change prediction to random for the step 
            after number of same actions reached the max_repetition.

            If partial_fit is set, do partial fit on the classifier if the 
            decision resulted in a reward.
        
        Keyword Arguments:
            render {bool} -- If True render the environment (default: {False})
            max_iter {int} -- If max_iter is set, use it as max number of iterations
                                insted of self.max_iter (default: {None})
        """
        if not max_iter:
            max_iter = self.max_iter

        next_obs = self.env.reset()
        next_action = None

        if self.max_repetition:
            current_repetition = 0

        for t in range(max_iter):
            obs = next_obs
            # get new prediction
            prev_action, next_action = next_action, self.predict(obs.reshape(1, -1))[0]

            #render environment
            if render:
                self.env.render()

            # change prediction to random if max_repetition is set and reached
            # only do this if reneding is disabled (we are probably training)
            elif self.max_repetition:
                current_repetition = current_repetition + 1 if next_action == prev_action else 0
                if current_repetition >= self.max_repetition:
                    next_action = self.env.action_space.sample()
                    current_repetition = current_repetition if next_action == prev_action else 0

            # get new observation
            next_obs, _, done, _ = self.env.step(next_action)
            if done:
                break

            # do partial fit
            if self.do_partial_fit and not render:
                self.partial_fit(
                    np.array([obs]),
                    np.array([next_action])
                )
        
        if render:
            self.env.close()
        else:
            # total reward = number of iterations classifier survived
            self.reward = t



# %%
