import random
from deap import base, creator, tools, algorithms

# Define the evaluation function (minimize a simple mathematical function)
def eval_func(individual):
    # Example evaluation function (minimize a quadratic function)
    return sum(x ** 2 for x in individual),

# DEAP setup
# Creating Types
# FitnessMin class - as we want to minimize the function. hence we set weights as -1.0
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# The Individual class in DEAP is a custom data structure (usually a list) that represents a solution to your optimization problem, with an extra .fitness attribute attached.
creator.create("Individual", list, fitness=creator.FitnessMin)


# Initialize toolbox
# this is where we add all the building blocks of the genetic algo (like central repo)
toolbox = base.Toolbox()

# individuals - represents a single solution or candidate in the search space. list or array of values that encode a solution to the problem being optimized.
# attributes - genes or features, are the individual elements that make up an individual.
# Define attributes and individuals
toolbox.register("attr_float", random.uniform, -5.0, 5.0)  
# create an individual in the DEAP toolbox
# individual - name given to registered method
# initCycle - method used to initialize the individual. apply a list of method to generate the individual
# creator.Individual - type of individual to be created
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)  
# to create a population of individuals
# population - name given to registered method
# initRepeat - function repeats a given function to generate a list of objects
# list - type of container that will hold the individuals
# toolbox.individual - method used to generate each individual in the population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Evaluation function and genetic operators
# Genetic operators
# evaluate - calculates the fitness of an individual. takes an individual as input and returns a fitness value. to determine how good or bad an individual is
toolbox.register("evaluate", eval_func)
# mate - combines two individuals to produce offspring. takes two individuals as input and returns a new individual
toolbox.register("mate", tools.cxBlend, alpha=0.5)
# mutate - introduce random changes to an individual genome
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
# select - chooses individuals to reproduce
toolbox.register("select", tools.selTournament, tournsize=3)


# Create population
population = toolbox.population(n=50)


# Genetic Algorithm parameters
generations = 20


# Run the algorithm
for gen in range(generations):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# Get the best individual after generations
best_ind = tools.selBest(population, k=1)[0]
best_fitness = best_ind.fitness.values[0]

print("Best individual:", best_ind)
print("Best fitness:", best_fitness)