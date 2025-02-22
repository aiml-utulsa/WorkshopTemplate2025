import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

policy = np.array(
    [
        [0.003, 0.003, 0.003, 0.991],
        [0.003, 0.003, 0.991, 0.003],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
    ]
)


env = gym.make("FrozenLake-v1", render_mode="human", desc=["SF", "HG"])
obs, info = env.reset()
gamma = 0.99
qtable = np.zeros((4, 4))
vtable = np.zeros(4)

# Q table
# [0,0,0,0]
# [0,0,0,0]
# [0,0,0,0]
# [0,0,0,0]
# V table
# [0,0,0,0]
#

q_counts = np.zeros((4, 4))
v_counts = np.zeros(4)
learning_rate = 0.02

obsarr = []
actarr = []
rewarr = []
print(obs)
for i in range(100000):
    action = np.random.multinomial(1, policy[obs, :]).argmax()
    obsarr.append(obs)
    actarr.append(action)

    obs, reward, terminated, truncated, info = env.step(action)
    rewarr.append(reward)
    if terminated or truncated:
        q_active = np.zeros((4, 4))
        v_active = np.zeros(4)
        ep_return = 0
        disc_pow = 0
        q_counts[obs, :] += 1  # (1 - learning_rate) * q_counts[obs, :] + learning_rate
        v_counts[obs] += 1  # (1 - learning_rate) * v_counts[obs] + learning_rate

        for t in range(len(obsarr) - 1, -1, -1):
            ep_return = rewarr[t] + ep_return * (gamma**disc_pow)
            disc_pow += 1

            # if q_active[obsarr[t], actarr[t]] == 0:
            q_active[obsarr[t], actarr[t]] += 1
            qtable[obsarr[t], actarr[t]] += ep_return
            q_counts[obsarr[t], actarr[t]] += 1

            # if v_active[obsarr[t]] == 0:
            v_active[obsarr[t]] += 1
            vtable[obsarr[t]] += ep_return
            v_counts[obsarr[t]] += 1
        obs, info = env.reset()
        obsarr = []
        actarr = []
        rewarr = []
env.close()

print("Q Values:")
print(qtable / q_counts)
print("V Values:")
print(vtable / v_counts)
print("Average of Q Values:")
print((qtable / q_counts * policy).sum(axis=1))
