# ANN
Evolutionary ANN to play Games

## File explanation

*neural_network.py* holds the classifier class. This class needs to have run_simulation method based on which fitness for the classifier is calculated.
This method can also be used to render the classifier on the screen.


*dna.py* contains DNA class. This class has methods like crossover and mutation as well as a main method that dictates how crossover and mutation are applied. The reason this is in a separate class is because we may want to use some different methods for achieving evolution and then it might be convenient to simply replace this class with some other.


*genetic_algorithm.py* contains GA class. This class has control over entire population of classifiers. How many there are, how to go from generation to generation, and when to stop. It is also responsible for parent selection process.

*run.py* contains run function. Simply run the algorithm with provided parameters. 

*extra.py* contains script for running the algorithm with many combinations of configurations and stores the results in  *data/results.csv*


## Resuls

