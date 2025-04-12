import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


from Agent import Agent
import random
import numpy as np
import matplotlib.pyplot as plt


class Mem_Buffer:
    def __init__(self, max_size=10000, obs_dim=13 * 13 * 5, action_dim=21):
        self.max_size = max_size
        self.states = np.zeros((10000, obs_dim))
        self.states_ = np.zeros((10000, obs_dim))
        self.actions = np.zeros((10000, 1), dtype=np.int64)
        self.rewards = np.zeros((10000))
        self.dones = np.zeros((10000))
        self.current_step = 0
        self.max_step = 0

    def add(self, state, action, reward, state_, done):
        self.states[self.current_step] = state
        self.states_[self.current_step] = state_
        self.actions[self.current_step] = action
        self.rewards[self.current_step] = reward
        self.dones[self.current_step] = done
        self.current_step += 1
        if self.current_step >= self.max_size:
            self.max_step = self.max_size
            self.current_step = 0
        if self.max_step < self.max_size:
            self.max_step = self.current_step

    def get(self, index):
        return (
            self.states[index],
            self.actions[index],
            self.rewards[index],
            self.states_[index],
            self.dones[index],
        )

    def sample(self, batch_size):
        idx = np.random.choice(self.max_step, batch_size, replace=False)
        return self.get(idx)

    def __len__(self):
        return self.max_step


class Q_network(nn.Module):
    def __init__(
        self,
        obs_dim,
        discrete_action_dim,
        hidden_dim=8,
        device="cuda:0",
        eval_mode=False,
        n_agents=2,
    ):
        super(Q_network, self).__init__()
        self.device = device
        self.l1 = nn.Linear(obs_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, discrete_action_dim)
        self.to(device)
        self.device = device

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class Q_Agent(Agent):
    def __init__(
        self, obs_dim=96 * 96 * 3, action_dim=21, hidden_dim=32, device="cuda:0"
    ):
        self.Q_net = Q_network(
            obs_dim=obs_dim,  # from MAgent2 environment or racing environment
            discrete_action_dim=action_dim,  # from MAgent2 environment or racing environment
            hidden_dim=hidden_dim,
            device="cuda:0",
        )
        self.discrete_dim = action_dim
        self.device = device
        self.optimizer = Adam(self.Q_net.parameters(), lr=3e-4)

    def transform_obs(self, obs):
        # print(obs.shape)
        # plt.imshow(obs)
        # plt.show()
        new_obs = np.zeros([16, 16, 3])
        for i in range(16):
            for j in range(16):
                for z in range(3):
                    new_obs[i][j][z] = (
                        np.mean(obs[i * 6 : (i + 1) * 6, j * 6 : (j + 1) * 6, z]) / 255
                    )

        # plt.imshow(new_obs)
        # plt.show()
        return new_obs

    def take_action(self, observation, id=0):
        if random.random() < 0.05:
            return random.randint(0, self.discrete_dim - 1)
        with torch.no_grad():
            torch_obs = torch.tensor(observation.flatten(), dtype=torch.float32).to(
                self.Q_net.device
            )
            q_values = self.Q_net(torch_obs)
            action = torch.argmax(q_values).cpu().item()

        return action

    def save(self, checkpoint_path):
        torch.save(self.Q_net.state_dict(), checkpoint_path + "Q_net_test")
        print("Saved model to", checkpoint_path)

    def load(self, checkpoint_path):
        self.Q_net.load_state_dict(
            torch.load(checkpoint_path + "Q_net_test", weights_only=True)
        )
