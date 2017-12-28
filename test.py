import numpy as np
from collections import Counter

all_such_actions = [[[0,1],[1,1]], [[2,1],[1,1]], [[7,1],[1,3]]]
def x(actionlist, pos):
    b = []
    c = map(lambda action: b.append(action) if action[1] == pos else 0, actionlist)
    return b
print(x(all_such_actions, [1,3]))
h = [1,3]
print(h == [1,3])