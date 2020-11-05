import Reporter
import numpy as np

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

		while( yourConvergenceTestsHere ):

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
