import numpy as np
from collections import Counter

x = np.ones((10,10))
b = np.where(x[0,0:10] == 1)
print(x)
print(x[0,0:10][b[0]])
for i in np.random.choice([0, 1, 2, 2, 2, 3, 3, 10, 11, 11], 10, replace=False):
    print("Ok")
