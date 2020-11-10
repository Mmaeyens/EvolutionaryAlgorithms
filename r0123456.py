import Reporter
import numpy as np
import random
import math
import Initialization
import Recombination


def length(individual: np.array, distance_matrix: np.array) -> float:
    distance = 0
    size = distance_matrix.shape[0]
    for i in range(size - 1):
        distance += distance_matrix[individual[i]][individual[i + 1]]
    distance += distance_matrix[individual[-1]][individual[0]]
    return distance


def elimination(population: np.array, offspring: np.array, distance_matrix: np.array, population_size: int) -> np.array:
    combined = np.concatenate((population, offspring), axis=0)
    combined = list(combined.astype(int))
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
    subset_size = 0

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

    '''
    possible alternative:
    while random.random() < alpha:
        first_index = random.randrange(individual_size)
        second_index = random.randrange(individual_size)
        temp = individual[first_index]
        individual[first_index] = individual[second_index]
        individual[second_index] = temp
    return individual
    '''

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
    random_selection.sort(key=lambda individual: length(individual, distance_matrix))
    return random_selection[0]


# Modify the class name to match your student number.
class r0123456:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename, population_size, its, recom_its, k,alpha):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        population = Initialization.initialize(population_size,distanceMatrix.shape[0])
        i = 0

        while (its > i):

            # Your code here.
            offspring = np.zeros([2*recom_its,distanceMatrix.shape[0]])
            # Recombination
            for j in range(0,2*recom_its,2):
                parent1 = selection(population,k, distanceMatrix)
                parent2 = selection(population,k, distanceMatrix)
                child1,child2 = Recombination.PMX(parent1,parent2)
                offspring[j] = child1
                offspring[j+1] = child2

            # Mutation
            for j in range(len(offspring)):
                offspring[j] =mutation(offspring[j],alpha)

            for j in range(len(population)):
                population[j] =mutation(population[j],alpha)

            # Elimination
            population = elimination(population,offspring,distanceMatrix,population_size)
            print("Score iteration {}".format(i),length(population[0],distanceMatrix))
            
            bestSolution = population[0]
            bestObjective = length(bestSolution, distanceMatrix)
            meanObjective = np.average([length(individual, distanceMatrix) for individual in population])
            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            #timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            #if timeLeft < 0:
            #    break
            i+=1

        # Your code here.
        return 0


TSP = r0123456()
TSP.optimize("tour29.csv",50,500,25,10,0.5)
