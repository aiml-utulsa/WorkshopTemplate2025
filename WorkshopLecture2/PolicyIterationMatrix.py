import numpy as np
import pandas as pd


Q_table = np.zeros((4, 4))
T_matrix = np.array(
    [
        [
            [0.66667, 0.00000, 0.33333, 0.00000, 0.00000],
            [0.33333, 0.33333, 0.33333, 0.00000, 0.00000],
            [0.33333, 0.33333, 0.33333, 0.00000, 0.00000],
            [0.66667, 0.33333, 0.00000, 0.00000, 0.00000],
        ],
        [
            [0.33333, 0.33333, 0.00000, 0.33333, 0.33333],
            [0.33333, 0.33333, 0.00000, 0.33333, 0.33333],
            [0.00000, 0.66667, 0.00000, 0.33333, 0.33333],
            [0.33333, 0.66667, 0.00000, 0.00000, 0.00000],
        ],
        [
            [0.00000, 0.00000, 1.00000, 0.00000, 0.00000],
            [0.00000, 0.00000, 1.00000, 0.00000, 0.00000],
            [0.00000, 0.00000, 1.00000, 0.00000, 0.00000],
            [0.00000, 0.00000, 1.00000, 0.00000, 0.00000],
        ],
        [
            [0.00000, 0.00000, 1.00000, 1.00000, 0.00000],
            [0.00000, 0.00000, 1.00000, 1.00000, 0.00000],
            [0.00000, 0.00000, 1.00000, 1.00000, 0.00000],
            [0.00000, 0.00000, 1.00000, 1.00000, 0.00000],
        ],
    ]
)
#        0       1       2       3
actions = ["left", "down", "right", "up"]
policy = [2, 1, 2, 2]
reward_index = 4

for iteration in range(30):
    print(Q_table)
    Q_table_new = np.copy(Q_table)
    for state in range(4):
        # for action in range(4):
        action = policy[state]
        q = 0
        r = T_matrix[state, action, reward_index]

        q = q + r
        for next_state in range(4):
            p_next = T_matrix[state, action, next_state]
            q_next = Q_table[next_state, policy[next_state]]
            q = q + p_next * q_next

        Q_table_new[state, action] = q

    Q_table = np.copy(Q_table_new)
    input()
