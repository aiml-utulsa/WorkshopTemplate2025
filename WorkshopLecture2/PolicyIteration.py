import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd

transition_table = pd.read_csv("transition_matrix.csv")
Q_table = pd.read_csv("Q_table.csv")
Q_table["action"] = Q_table["action"].str.strip()
Q_table["action"] = Q_table["action"].str.replace("'", "")
Q_table["state"] = Q_table["state"].astype("int")


print(transition_table.head())
print(Q_table.head())
print(Q_table["action"].unique())

actions = ["left", "down", "right", "up"]
policy = ["down", "down", "down", "down"]


# Lets evaluate our policy
for i in range(10):
    print(Q_table)
    for state in range(4):
        q = 0
        r = transition_table[
            (transition_table["state"] == state)
            & (transition_table["action"] == policy[state])
        ]["r_prob"].values[0]

        for next_state in range(4):
            prob = transition_table[
                (transition_table["state"] == state)
                & (transition_table["action"] == policy[state])
            ][str(next_state)].values[0]
            q += (
                prob
                * Q_table[
                    (Q_table["state"] == next_state)
                    & (Q_table["action"] == policy[next_state])
                ]["Q"].values[0]
            )

        q += r
        Q_table.loc[
            (Q_table["action"] == policy[state]) & (Q_table["state"] == state), "Q"
        ] = q
    input("Next iteration?")
# env = gym.make("FrozenLake-v1", render_mode="human", desc=["SF", "HG"])
# obs, info = env.reset()
