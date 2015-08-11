import time
import numpy as np

grid = np.zeros((500,500))

start_time = time.clock()
for i in range(500):
    for j in range(500):
        grid[i,j] = 1

print time.clock() - start_time, "seconds"

grid = np.zeros((500,500))

start_time = time.clock()

grid[:,:] = 1

print time.clock() - start_time, "seconds"
