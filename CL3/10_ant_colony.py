import numpy as np


# Define the problem - set of cities and distances
# each row represent city's [x,y] co-ordinates
cities = np.array([[0, 0], [1, 2], [3, 1], [5, 3], [2, 4]])
num_cities = len(cities)


# Define parameters
# set the number of ants in colony
num_ants = 5
# 100 iterations
num_iterations = 100
# importance of trails in decision making. high alpha = ants pay more attention to existing pheromone
alpha = 1  # Pheromone factor
# set imp of distance in decision making. high beta = ants prefer shorter paths
beta = 2  # Distance factor
# evaporation rate after each iteration. low values maintain pheromone longer
rho = 0.1  # Pheromone evaporation rate
# scales how much pheromone ants deposit on trails
Q = 1  # Pheromone deposit factor


# Initialize pheromone matrix
# Creates a matrix filled with 1's to represent initial pheromone levels between each pair of cities.
pheromone = np.ones((num_cities, num_cities))


# Define distance function
# calculate eucledian distance
def distance(city1, city2):
    return np.linalg.norm(city1 - city2)


# Initialize best tour
# Initializes variables to keep track of the best tour found and its distance.
# np.inf sets the initial best distance to infinity.
best_tour = None
best_distance = np.inf


# Perform iterations
for iteration in range(num_iterations):

    ant_tours = []
    tour_distances = []

    # Move ants
    for ant in range(num_ants):
        # randomly selects a starting city for current ant
        current_city = np.random.randint(num_cities)
        # initialize ants tour with starting city
        tour = [current_city]
        # initialize distance travel to 0
        distance_traveled = 0

        # continue untill all cities visited
        while len(tour) < num_cities:
            # Creates an empty list to store the probabilities of visiting each unvisited city.
            probabilities = []
            # Loops through all cities, considering only unvisited ones.
            for city in range(num_cities):
                if city not in tour:
                    # gets pher lvl on the path from current city to candidate city
                    pheromone_level = pheromone[current_city][city]
                    # calc dist between current and candidate city
                    dist = distance(cities[current_city], cities[city])
                    # Calculates the probability based on pheromone level and distance.
                    # Higher pheromone and shorter distance = higher probability.
                    prob = (pheromone_level**alpha) * ((1 / dist)**beta)
                    # Adds the city and its selection probability to the list.
                    probabilities.append((city, prob))

            # Converts the list to a NumPy array for easier manipulation.
            probabilities = np.array(probabilities)
            # Normalizes the probabilities so they sum to 1.
            probabilities[:, 1] /= np.sum(probabilities[:, 1])
            # Randomly selects the next city based on the calculated probabilities.
            next_city = np.random.choice(probabilities[:, 0],
                                         p=probabilities[:, 1])
            # Adds the selected city to the ant's tour.
            tour.append(int(next_city))
            # Adds the distance to the next city to the total distance traveled.
            distance_traveled += distance(cities[current_city],
                                          cities[int(next_city)])
            # Updates the current city to the newly selected city.
            current_city = int(next_city)

        # Adds the completed tour to the list of all ant tours.
        ant_tours.append(tour)
        # Adds the total distance of the tour to the list of all tour distances.
        tour_distances.append(distance_traveled)


    # Update pheromone levels
    # Evaporates some pheromone from all paths (reduces all values by 10%).
    pheromone *= (1 - rho)
    # Loops through each ant's tour to update pheromone levels.
    for i in range(num_ants):
        tour = ant_tours[i]
        # Deposits pheromone on each edge of the tour.
        # Shorter tours get more pheromone (1/distance).
        for j in range(num_cities - 1):
            pheromone[tour[j]][tour[j + 1]] += (Q / tour_distances[i])
        # Adds pheromone to the edge connecting the last city back to the first.
        # This completes the cycle for the Traveling Salesman Problem.
        pheromone[tour[-1]][tour[0]] += (Q / tour_distances[i])


    # Update best tour
    # Finds the index of the ant with the shortest tour in this iteration.
    min_distance_idx = np.argmin(tour_distances)
    # If this tour is better than the best so far, update the best tour and distance.
    if tour_distances[min_distance_idx] < best_distance:
        best_tour = ant_tours[min_distance_idx]
        best_distance = tour_distances[min_distance_idx]

print("Best tour:", best_tour)
print("Best distance:", best_distance)