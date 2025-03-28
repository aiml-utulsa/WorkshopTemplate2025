import gymnasium as gym
import numpy as np
from Agent import Agent
import random


def evaluate_agent(agent: Agent, env, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.take_action(obs, id=0)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        total_rewards.append(total_reward)
        print("Total reward: ", total_reward)
    return np.mean(total_rewards), np.std(total_rewards)


if __name__ == "__main__":
    from CompetitionAgents.singleAgent.Q_net_example import Q_agent
    from CompetitionAgents.randomAgent.rand_agent import Random_Agent

    agents = [Random_Agent(), Q_agent]
    comp_agent_folders = [
        "./CompetitionAgents/randomAgent/",
        "./CompetitionAgents/singleAgent/Q_net_example/",
    ]

    env = gym.make(
        "CarRacing-v2",  # v3 for linux
        render_mode="human",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=False,
    )

    for comp_agent in range(len(agents)):
        agents[comp_agent].load(comp_agent_folders[comp_agent])
        print("Loaded agent: ", comp_agent)
    for agent in agents:
        mean, std = evaluate_agent(agent, env, num_episodes=100)
        print(f"Agent: {agent.__class__.__name__}")
        print(f"Mean: {mean}, Std: {std}")
