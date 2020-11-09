import Reporter
import numpy as np
import random
import math


def length(individual: np.array, distance_matrix: np.array) -> float:
    distance = 0
    size = distance_matrix.shape[0]
    for i in range(size - 1):
        distance += distance_matrix[individual[i]][individual[i + 1]]
    distance += distance_matrix[individual[-1]][individual[0]]
    return distance


def elimination(population: np.array, offspring: np.array, distance_matrix: np.array, population_size: int) -> np.array:
    combined = list(np.concatenate((population, offspring), axis=0))
    combined.sort(key=lambda individual: length(individual, distance_matrix))
    new_population = combined[:population_size]
    return np.array(new_population)


def mutation(individual: np.array, alpha: float) -> np.array:
    '''
    swaps 2 subsets where the size of the subset is an integer size determined by alpha between 1 and len(individual)/2

    :param individual: numpy array, containing the order of visiting the locations
    :param alpha: float, chance to increase subset size by 1 each iteration
    :return:
    '''
    print("individual", individual)
    subset_size = 1

    individual_size = len(individual)

    # double size of individual since we are want to treat it as a continuous path without beginning or end
    individual = np.append(individual, individual)

    # determine size of individual maybe making a pdf could be better
    while random.random() < alpha:
        subset_size += 1
        if subset_size == math.floor(individual_size / 2):
            break

    first_index = random.randrange(individual_size)

    # determine position of swapping subset so it can't overlap
    offset = first_index + subset_size + random.randrange(individual_size - 2 * subset_size)

    temp = individual[first_index:first_index + subset_size].copy()
    individual[first_index:first_index + subset_size] = individual[offset:offset + subset_size]
    individual[offset:offset + subset_size] = temp

    return individual[first_index:first_index + individual_size]


def selection(population: np.array, k: int, distance_matrix: np.array):
    '''
    k tournament selection, selects k random individual then selects the best from that group
    :param population: numpy array, containing all the individuals
    :param k: int, initial size of random selected group
    :param distance_matrix: numpy array, necessary for fitness calculation
    :return:
    '''
    random_selection = random.choices(population, k=k)
    return random_selection.sort(key=lambda individual: length(individual, distance_matrix))[0]


# Modify the class name to match your student number.
class r0123456:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.

        while (yourConvergenceTestsHere):

            # Your code here.

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

        # Your code here.
        return 0


print(mutation(np.arange(9), 0.3))
