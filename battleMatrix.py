def get_battle_matrix():
    battleMatrix = dict()
    battleMatrix[1, 11] = -1
    battleMatrix[1, 1] = 0
    battleMatrix[1, 2] = -1
    battleMatrix[1, 3] = -1
    battleMatrix[1, 0] = 1
    battleMatrix[1, 10] = 1
    battleMatrix[2, 0] = 1
    battleMatrix[2, 11] = -1
    battleMatrix[2, 1] = 1
    battleMatrix[2, 2] = 0
    battleMatrix[2, 3] = -1
    battleMatrix[2, 10] = -1
    battleMatrix[3, 0] = 1
    battleMatrix[3, 11] = 1
    battleMatrix[3, 2] = 1
    battleMatrix[3, 3] = 0
    battleMatrix[3, 1] = 1
    battleMatrix[3, 10] = -1
    battleMatrix[10, 0] = 1
    battleMatrix[10, 11] = -1
    battleMatrix[10, 1] = 1
    battleMatrix[10, 2] = 1
    battleMatrix[10, 3] = 1
    battleMatrix[10, 10] = 0
    return battleMatrix