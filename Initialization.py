import numpy as np

def initialize(population_size,number_of_nodes):
    print('Initialization of a population with population_size '+str(population_size)+'...')
    population = []
    for i in range(population_size):
        individual = np.arange(number_of_nodes)
        np.random.shuffle(individual)
        population.append(individual)
    population = np.asarray(population)
    print('Initialization of the population is completed')
    return population

def recombination():
    print('Recombination phase')

def main():
    #file = open('tour29.csv')
    #distanceMatrix = np.loadtxt(file, delimiter=",")
    #file.close()
    population = initialize(5,5)
    print(population)

main()