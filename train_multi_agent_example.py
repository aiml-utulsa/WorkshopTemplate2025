from CompetitionAgents.marlAgent.Q_net_example.Q_agent import Q_Agent
from CompetitionAgents.marlAgent.rand_agent import Random_Agent
from CompetitionAgents.marlAgent.Q_net_example.Q_agent import Mem_Buffer
from torch import nn
import numpy as np
from Agent import Agent
from magent2.environments import battle_v4
import torch


def train_loop(
    num_episodes,
    agent1: Agent,
    agent2: Agent,
    env,
    render=False,
    comp_agent_folders=None,
):
    observations = env.reset()
    max_agents = env.num_agents
    mem_buffers = []
    episode_reward_history = []

    epsilon = 1.0  # exploration rate
    epsilon_decay = 0.97  # decay rate for exploration probability
    for i in range(max_agents):
        mem_buffers.append(Mem_Buffer(max_size=10000))

    for episode in range(num_episodes):
        epsilon = max(0.05, epsilon * epsilon_decay)  # decay the exploration rate
        print(epsilon)
        tot_reward = 0
        observations = env.reset()
        max_agents = env.num_agents
        agent_nums = {}
        step = 0
        for i, agent_name in enumerate(env.agents):
            agent_nums[agent_name] = i
        while env.agents:
            actions = {}
            for agent_name in env.agents:
                if agent_name[0:3] == "red":
                    if np.random.rand() >= epsilon:
                        actions[agent_name] = agent1.take_action(
                            observations[agent_name], id=agent_nums[agent_name]
                        )
                    else:
                        # training partner is the random agent so we take a random action
                        actions[agent_name] = agent2.take_action(
                            observations[agent_name],
                            id=agent_nums[agent_name],
                        )

                elif agent_name[0:4] == "blue":
                    actions[agent_name] = agent2.take_action(
                        observations[agent_name],
                        id=agent_nums[agent_name] - (max_agents // 2),
                    )
            new_observations, rewards, terminations, truncations, info = env.step(
                actions
            )
            for agent_name in env.agents:
                if agent_name[0:3] == "red":
                    tot_reward += rewards[agent_name]

            for agent_name in env.agents:
                if agent_name[0:3] == "red":
                    mem_buffers[agent_nums[agent_name]].add(
                        observations[agent_name].flatten(),
                        actions[agent_name],
                        rewards[agent_name],
                        new_observations[agent_name].flatten(),
                        float(terminations[agent_name]),
                    )

            # every 16 steps we train the agents
            if step % 16 == 0:
                for buffer in mem_buffers:
                    if len(buffer) > 128:
                        states, actions, rewards, states_, dones = buffer.sample(128)
                        states = torch.tensor(states).float().to(agent1.device)
                        actions = torch.tensor(actions).to(agent1.device)
                        rewards = torch.tensor(rewards).float().to(agent1.device)
                        states_ = torch.tensor(states_).float().to(agent1.device)
                        dones = torch.tensor(dones).to(agent1.device)

                        q_values = agent1.Q_net(
                            states
                        )  # get q values for each action for current states
                        q_values = torch.gather(
                            q_values, dim=-1, index=actions.long()
                        ).squeeze(
                            -1
                        )  # Get the q values for the action we chose
                        with torch.no_grad():
                            next_q_values = torch.max(
                                agent1.Q_net(states_), dim=-1
                            ).values
                            targets = rewards + 0.99 * (1 - dones) * next_q_values

                        loss = ((q_values - targets) ** 2).mean()  # calculate loss
                        agent1.optimizer.zero_grad()
                        loss.backward()  # backpropagate loss
                        agent1.optimizer.step()  # update weights

            observations = new_observations.copy()

        step += 1
        print(tot_reward)
        episode_reward_history.append(tot_reward)
        if comp_agent_folders is not None:
            agent1.save(comp_agent_folders[0])
    return episode_reward_history


if __name__ == "__main__":
    agents = [Q_Agent(), Random_Agent()]
    comp_agent_folders = [
        "./CompetitionAgents/marlAgent/Q_net_example/",  # Q agent
    ]

    agents[0].save(comp_agent_folders[0])  # save the agent to the folder
    agents[0].load(comp_agent_folders[0])

    env = battle_v4.parallel_env(map_size=16, render_mode="human")
    train_loop(
        100,
        agents[0],
        agents[1],
        env,
        render=False,
        comp_agent_folders=comp_agent_folders,
    )
