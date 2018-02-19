import numpy as np
import matplotlib.pyplot as plt


N = 6
win_flag = (641 + 260 + 491 + 337 + 210, 170 + 305 +  + 632, 30, 35, 614 + 665 + 820 +719 )
win_moves = (132 + 80 + 114 + 150 + 131, 32 + 214 +  + 187, 34, 20, 118 + 120 + 100 +79 )
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, win_flag, width)
p2 = plt.bar(ind, win_moves, width,
             bottom=win_flag)

plt.ylabel('Number of total wins in all simulations')
plt.title('Scores by flag capture and won by moves')
plt.xticks(ind, ('MM', 'MC', 'Q-MM', 'Q-MC', 'DQN', "OMM"))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Flag', 'Moves'))

plt.show()