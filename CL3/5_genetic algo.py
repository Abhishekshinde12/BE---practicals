# DEAP - here we either max / min the function

import random
from deap import base, creator, tools, algorithms

# Define genetic algorithm parameters
POPULATION_SIZE = 10
GENERATIONS = 5


# Define evaluation function
def evaluate(individual):
    # Here 'individual' represents the parameters for the neural network
    # Replace this with your actual evaluation function that trains the neural network
    # and evaluates its performance
    # Return a fitness value (here, a random number is used as an example)
    return random.random()


# Creating Types
# FitnessMin class - as we want to minimize the function. hence we set weights as -1.0
creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
# The Individual class in DEAP is a custom data structure (usually a list) that represents a solution to your optimization problem, with an extra .fitness attribute attached.
creator.create("Individual", list, fitness=creator.FitnessMin)

# Initialize toolbox
# this is where we add all the building blocks of the genetic algo (like central repo)
toolbox = base.Toolbox()

# individuals - represents a single solution or candidate in the search space. list or array of values that encode a solution to the problem being optimized.
# attributes - genes or features, are the individual elements that make up an individual.
# Define attributes and individuals

# Example: no. of neurons
toolbox.register("attr_neurons", random.uniform, 1, 100)
# Example: no. of layers
toolbox.register("attr_layers", random.uniform, 1, 5)
# create an individual in the DEAP toolbox
# individual - name given to registered method
# initCycle - method used to initialize the individual. apply a list of method to generate the individual
# creator.Individual - type of individual to be created
# (toolbox.attr_neurons, toolbox.attr_layers) - list of methods to be applied to generate the individual. attr_neurons and attr_layers generate a no. of nuerons and layres.
# n = 1 ==> no. of times the method should be applied to generate the individual
# as 2 attributes - hence the resulting individual will be a list of 2 values.
toolbox.register("individual",
                 tools.initCycle,
                 creator.Individual,
                 (toolbox.attr_neurons, toolbox.attr_layers),
                 n=1)
# to create a population of individuals
# population - name given to registered method
# initRepeat - function repeats a given function to generate a list of objects
# list - type of container that will hold the individuals
# toolbox.individual - method used to generate each individual in the population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic operators
# evaluate - calculates the fitness of an individual. takes an individual as input and returns a fitness value. to determine how good or bad an individual is
toolbox.register("evaluate", evaluate)
# mate - combines two individuals to produce offspring. takes two individuals as input and returns a new individual
toolbox.register("mate", tools.cxTwoPoint)
# mutate - introduce random changes to an individual genome
toolbox.register("mutate", tools.mutUniformInt, low=1, up=100, indpb=0.2)
# select - chooses individuals to reproduce
toolbox.register("select", tools.selTournament, tournsize=3)


# Create initial population
# Each individual is a random NN config (no. of neurons and layers)
population = toolbox.population(n=POPULATION_SIZE)


# Run the genetic algorithm
for gen in range(GENERATIONS):
  offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
  fitnesses = toolbox.map(toolbox.evaluate, offspring)
  for ind, fit in zip(offspring, fitnesses):
    ind.fitness.values = fit
  population = toolbox.select(offspring, k=len(population))

# Get the best individual from the final population
best_individual = tools.selBest(population, k=1)[0]
best_params = best_individual

# Print the best parameters found
print("Best Parameters:", best_params)