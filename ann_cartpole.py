from sklearn.neural_network import MLPClassifier
import gym, numpy as np, time, math, random

def initialize_ann(n,env):
    """
    Creating generation of n ANNs, and partially training them using random data from the CartPole environment.
    :param n: Number of ANNs in a generation.
    :param env: The CartPole environment.
    :return: Array of partially trained ANNs
    """
    #list of ANNs
    ann= []
    #generating ANNs
    for i in range(n):
        ann.append(MLPClassifier(batch_size=1, max_iter=1, solver='sgd', activation='relu', learning_rate='invscaling', hidden_layer_sizes=hlayer_size, random_state=1))
    ann = np.array(ann) #converting list to array
    #partial training
    for i in range(n):
        ann[i].partial_fit(np.array([env.observation_space.sample()]),np.array([env.action_space.sample()]),classes=np.arange(env.action_space.n))
    return ann

def predict_step(env, ann, prev_action, repeated_action):
    """
    Recursive function that predicts the next action for the ANN to do to balance the pole in the CartPole-environment.
    :param env: The CartPole environment.
    :param ann: The artificial neural network to be used for predicting next action.
    :param prev_action: The previous action taken, used to check for more than 5 repeated actions.
    :param repeated_action: How many times the previous action has been repeated. 
    :return: Reward 
    """
    #small time delay to slow down rendering
    #time.sleep(0.03)
    #performing action from previous step in recursive calls
    observation, reward, done, info = env.step(prev_action)
    #env.render()

    #predicting next step
    next_action = ann.predict(observation.reshape(1,-1))
    #checking if predicted action is same as previous 5 actions
    if next_action == prev_action:
        if repeated_action >= 4: #if action is same as previous 5, random sample
            action = env.action_space.sample()
        else:
            action = next_action
            repeated_action+=1
    else: #different action, reset repeated_Action counter
        repeated_action=0
        action = next_action  
    #converting from array if actions is an array
    if isinstance(action,int):
        pass
    else:
        action=action[0]
    #checking if done
    if done:
        return 0
    #recursive call
    else:
        #doing a partial fit with observations from previous action and the predicted next action 
        #ann.partial_fit(np.array([observation]),np.array([action]),classes=np.arange(env.action_space.n))
        #recursive call using the predicted action
        r_reward = predict_step(env,ann,action,repeated_action)
    #adding reward of this step with rewards in recursive calls
    reward +=r_reward
    return reward

def simulate_generation(env, anns):
    """
    Runs simulations for one generation of ANNs in the CartPole environment.
    :param env: The CartPole environment to test the ANNs against.
    :param anns: The array of ANNs to be used.
    :return: Rewards for the different ANNs, and ANNs
    """

    rewards=np.zeros(n) #array for saving rewards

    #running simulation
    for i, x in enumerate(anns):
        time.sleep(0) #small time delay between each simulation
        env.reset()
        action = env.action_space.sample() #generating random action to start simulation
        rewards[i]=predict_step(env, x, action, 0) #starting recursion and simulation
    env.close()


    print('The average reward was {}.'.format(np.average(rewards)))
    print('The maximum reward was {} for ANN nr {}.'.format(np.max(rewards),(np.argmax(rewards)+1)))

    return rewards, anns

def non_uniform(rewards):
    """
    Taking probabilities and calculating a cumulative sum of these in an array.
    Using a uniformly distribuited number between 0 and 1 to choose one of the elements, by index.
    :param rewards: The probabilities of the array elements. Element 0 has the probability in prob[0] and so on.
    :return: Index of the chosen array element based on the probabilities.
    """
    #calculating probabilities based on rewards
    prob=rewards/(np.sum(rewards))
    dist = np.cumsum(prob)
    #generating uniformly distributed random number between 0 and 1
    num = random.random()
    #finding the index of the cumulative probability range that the random number is in
    index = len(dist[dist<num])

    return index

def pick_parent_ANNs(anns, rewards):
    """
    Picks two different parent ANNs based on the probabilities calculated from the rewards they received.
    :param anns: The array of ANNs to choose from.
    :param rewards: The rewards for the ANNs from the simulation.
    :return: Two ANNs to be parent ANNs.
    """
    #finding index of first parent using rewards to calculate probabilities and picking based on these
    idx1 = non_uniform(rewards)
    ann1 = anns[idx1] #first parent ANN
    #finding index of next ANN by doing the same, but excluding the 
    idx2 = non_uniform(np.setdiff1d(rewards,rewards[idx1]))
    #finding parent 2 from list
    if idx2 >= idx1: #correcting index for 
        idx2 = idx2+1
    ann2 = anns[idx2]
    return ann1, ann2

def mutation(mut_rate, ann1, ann2):
    """
    Checking whether there is to be a mutation in weights or biases spearately. 
    Finding random spot for mutation in the weights and/or biases.
    Performing mutation.

    As described in assignment text, separately for weights and biases.

    :param mut_rate: Rate of mutation
    :param ann1: First ANN to possibly perform mutation on.
    :param ann2: Second ANN to possibly perform mutation on.
    :return: None
    """

    #for weights
    mut = non_uniform(np.array([(1-mut_rate),mut_rate])) #using the nonuniform function, will return index 0 or 1, correlating to no mutation (0) and mutation (1)
    if mut == 1:
        out_in = random.randint(0,1) #weights from input (0) or to output (1)
        if out_in == 0: #from inputs
            row = random.randint(0,3)
            col = random.randint(0,3)
            #holding copy
            copy_2 = np.copy(ann2.coefs_[0][row][col])
            #mutating by switching values
            ann2.coefs_[0][row][col]=ann1.coefs_[0][row][col]
            ann1.coefs_[0][row][col]=copy_2
        else: #to outputs
            row = random.randint(0,3)
            #holding copy
            copy_2 = np.copy(ann2.coefs_[0][row])
            #mutating by switching values
            ann2.coefs_[0][row]=ann1.coefs_[0][row]
            ann1.coefs_[0][row]=copy_2

    #for bias
    mut = non_uniform(np.array([(1-mut_rate),mut_rate])) #using the nonuniform function, will return index 0 or 1, correlating to no mutation (0) and mutation (1)
    if mut == 1:
        out_in = random.randint(0,1) #biases from input (0) or to output (1)
        if out_in == 0: #from inputs
            row = random.randint(0,3)
            #holding copy
            copy_2 = np.copy(ann2.intercepts_[0][row])
            #mutating by switching values
            ann2.intercepts_[0][row]=ann1.intercepts_[0][row]
            ann1.intercepts_[0][row]=copy_2
        else: #to outputs
            #holding copy
            copy_2 = np.copy(ann2.intercepts_[1])
            #mutating by switching values
            ann2.intercepts_[1]=ann1.intercepts_[1]
            ann1.intercepts_[1]=copy_2

    return None

def crossover_children(env, ann1, ann2, mut_rate):
    """
    Making two new children ANNs from two parent ANNs by crossover. 
    
    THERE ARE DIFFERENT CROSSOVER ALGORITHMS TO CHOOSE FROM.
    HERE IMPLEMENTED WHAT IS SUGGESTED, WHERE APPLICABLE, CHOOSE 1 RANDOM INDEX, 
    THEN SPLIT THE DATA FOR THE PARENTS ON THIS INDEX AND ASSIGN THE DIFFERENT BITS TO THE CHILD.

    SPLIT ON INDIVIDUAL WEIGHT WHERE APPLICABLE, NOT ARRAY, OR TOTAL NUMBER OF WEIGHTS.

    :param env: The CartPole environment.
    :param ann1: Parent ANN 1.
    :param ann2: Parent ANN 2.
    :return: Two child ANNs.
    """

    #performing possible mutation on parents
    mutation(mut_rate,ann1,ann2)

    c_ann1 = MLPClassifier(batch_size=1, max_iter=1, solver='sgd', activation='relu', learning_rate='invscaling', hidden_layer_sizes=hlayer_size, random_state=1)
    c_ann2 = MLPClassifier(batch_size=1, max_iter=1, solver='sgd', activation='relu', learning_rate='invscaling', hidden_layer_sizes=hlayer_size, random_state=1)

    #running partial fit to set up the weights and intercepts
    c_ann1.partial_fit(np.array([env.observation_space.sample()]),np.array([env.action_space.sample()]),classes=np.arange(env.action_space.n))
    c_ann2.partial_fit(np.array([env.observation_space.sample()]),np.array([env.action_space.sample()]),classes=np.arange(env.action_space.n))
    
    ridx_w = random.randint(0,3)
    #assigning weights child ANN 1
    for i in range(len(ann1.coefs_[0])):
        c_ann1.coefs_[0][i][:ridx_w] = ann1.coefs_[0][i][:ridx_w]
        c_ann1.coefs_[0][i][ridx_w:] = ann2.coefs_[0][i][ridx_w:]

    c_ann1.coefs_[1][:ridx_w] = ann1.coefs_[1][:ridx_w]
    c_ann1.coefs_[1][ridx_w:] = ann2.coefs_[1][ridx_w:]

    #assigning weights child ANN 2
    for i in range(len(ann1.coefs_[0])):
        c_ann2.coefs_[0][i][:ridx_w] = ann2.coefs_[0][i][:ridx_w]
        c_ann2.coefs_[0][i][ridx_w:] = ann1.coefs_[0][i][ridx_w:]

    c_ann2.coefs_[1][:ridx_w] = ann2.coefs_[1][:ridx_w]
    c_ann2.coefs_[1][ridx_w:] = ann1.coefs_[1][ridx_w:]

    ridx_b = random.randint(0,3)
    #assigning biases child ANN 1
    c_ann1.intercepts_[0][:ridx_b] = ann1.intercepts_[0][:ridx_b]
    c_ann1.intercepts_[0][ridx_b:] = ann2.intercepts_[0][ridx_b:]
    c_ann1.intercepts_[1] = ann1.intercepts_[1] #simply assigning this since there is only 1 value
    
    #assigning biases child ANN 2
    c_ann2.intercepts_[0][:ridx_b] = ann2.intercepts_[0][:ridx_b]
    c_ann2.intercepts_[0][ridx_b:] = ann1.intercepts_[0][ridx_b:]
    c_ann2.intercepts_[1] = ann2.intercepts_[1]#simply assigning this since there is only 1 value

    return c_ann1, c_ann2

def new_generation(env, anns, rewards, mut_rate):
    """
    Make a new generation of ANNs.
    :param anns: Array of ANNs from previous generation.
    :param rewards: ANN probabilities based on rewards. 
    :return: New generation of ANNs made from chosen parent ANNs.
    """ 
    #number of ANNs
    n=len(anns)

    child_anns = []
    #half of the total number of ANNs since each running of the loop, will give 2 children
    run_n = int((n/2)+0.5) #adding 0.5, rounding and converting to int to avoid any issues with float numbers
    for _ in range(run_n):
        ann1, ann2 = pick_parent_ANNs(anns,rewards)
        c_ann1, c_ann2 = crossover_children(env,ann1, ann2, mut_rate)
        child_anns.append(c_ann1)
        child_anns.append(c_ann2)
    
    #turning list to array
    child_anns = np.array(child_anns)

    return child_anns

def average_ann(env, all_ann, n, g):
    """
    Function calculates average weights and biases for all generations of ANNs.
    Then initializes a new ANN with these average weights and biases.
    :param env: The CartPole environment.
    :param all_ann: List of all ANNs from all generations.
    :param n: Number of ANNs in one generation.
    :param g: Number of generations.
    :return: The ANN with average weights and biases.
    """
    #constructing average weights and biases across all generations
    weights_in = all_ann[0][0].coefs_[0]
    weights_out = all_ann[0][0].coefs_[1]
    biases_in = all_ann[0][0].intercepts_[0]
    biases_out = all_ann[0][0].intercepts_[1]
    for i in range(g):
        for j in range(n):
            if i == 0 and j == 0:
                pass
            else:
                weights_in += all_ann[i][j].coefs_[0]
                weights_out += all_ann[i][j].coefs_[1]
                biases_in += all_ann[i][j].intercepts_[0]
                biases_out += all_ann[i][j].intercepts_[1]

    #finding average:
    weights_in=weights_in/(n*g)
    weights_out=weights_out/(n*g)
    biases_in=biases_in/(n*g)
    biases_out=biases_out/(n*g)

    #initializing average ANN
    average_ann = MLPClassifier(batch_size=1, max_iter=1, solver='sgd', activation='relu', learning_rate='invscaling', hidden_layer_sizes=hlayer_size, random_state=1)
    average_ann.partial_fit(np.array([env.observation_space.sample()]),np.array([env.action_space.sample()]),classes=np.arange(env.action_space.n))

    #assigning weights and biases
    average_ann.coefs_[0] = weights_in
    average_ann.coefs_[1] = weights_out
    average_ann.intercepts_[0] = biases_in
    average_ann.intercepts_[1] = biases_out

    return average_ann

def ea_ann_simulation(env, n, g, hlayer_size, mut_rate):
    """
    Function that develops multiple generations of ANNs to solve the CartPole environment problem,
    using crossover and mutation algorithms. 
    :param env: The CartPole environment
    :param n: Number of ANNs in each generation.
    :param g: Number of generations.
    :param hlayer_size: The size(s) of the hidden layer(s) in the ANNs.
    :param mut_rate: The mutation rate for the ANNs.
    :return: Best ANN in the final generation and the ANN with average weight and biases.
    """
    
    #Initializing ANNs
    ann=initialize_ann(n,env)

    #list for saving all ANNs
    all_ann = []
    all_rewards = []

    for i in range(g):
        print('\nGeneration {}:'.format(i+1))
        #Simulating one generation and storing rewards
        rewards, annP = simulate_generation(env, ann)
        all_ann.append(annP)
        all_rewards.append(rewards)
        ann = new_generation(env,annP,rewards,mut_rate)

    reward_max = 0
    max_idx = []
    for i, x in enumerate(all_rewards):
        if np.max(x) >= reward_max:
            reward_max = np.max(x)
            max_idx = np.array([i,np.argmax(x)])

    best_ann = all_ann[max_idx[0]][max_idx[1]]
    mean_ann = average_ann(env,all_ann,n,g)

    ann_compare = np.array([best_ann, mean_ann])

    print('Best ANN vs ANN with average weights and biases:')
    rewards, ann_compare = simulate_generation(env, ann_compare)

    return best_ann, mean_ann

env=gym.make('CartPole-v1') #initializing CartPol environment
env.reset()

n=50 #number of ANNs, has to be even number
assert (n%2)==0
g=20 #number of generations
mut_rate = 0.01 #mutation rate, between 0 and 1.
assert mut_rate**2 <= 1

hlayer_size = 4 #nodes in hidden layer

best_ann, all_ann = ea_ann_simulation(env,n,g,hlayer_size,mut_rate)

print(best_ann)

