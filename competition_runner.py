import gymnasium as gym
import numpy as np
from Agent import Agent


def evaluate_agent(agent: Agent, env, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.take_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards), np.std(total_rewards)


if __name__ == "__main__":
    from CompetitionAgents.timAgent.pid_agent import PID_Agent
    from CompetitionAgents.randomAgent.rand_agent import Random_Agent

    agents = [PID_Agent(), Random_Agent()]
    comp_agent_folders = [
        "./CompetitionAgents/timAgent/",
        "./CompetitionAgents/randomAgent/",
    ]

    env = gym.make("CartPole-v1")

    for comp_agent in range(len(agents)):
        agents[comp_agent].load(comp_agent_folders[comp_agent])

    for agent in agents:
        mean, std = evaluate_agent(agent, env)
        print(f"Agent: {agent.__class__.__name__}")
        print(f"Mean: {mean}, Std: {std}")
