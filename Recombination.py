import numpy as np
from datetime import datetime
import random


def PMX(parent1, parent2):
    child1 = parent1.copy()
    child2 = parent2.copy()
    size = min(len(child1), len(child2))
    p1, p2 = [0] * size, [0] * size

    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[child1[i]] = i
        p2[child2[i]] = i
    # Choose crossover points
    #print('size ' + str(size))
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    #print('slicing between ' + str(cxpoint1) + ' and ' + str(cxpoint2))
    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = child1[i]
        temp2 = child2[i]
        # Swap the matched value
        child1[i], child1[p1[temp2]] = temp2, temp1
        child2[i], child2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return child1, child2

def main():
    random.seed(datetime.now())
    print("start")
    path1 = np.array([4, 0, 3, 1, 2, 5])
    path2 = np.array([1, 5, 0, 4, 2, 3])
    child1, child2 = PMX(path1, path2)
    print(child1)
    print(child2)

main()
