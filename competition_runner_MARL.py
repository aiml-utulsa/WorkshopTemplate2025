import gymnasium as gym
import numpy as np
from Agent import Agent
from magent2.environments import battle_v4


def evaluate_agent(agent1: Agent, agent2: Agent, env, num_episodes=10):
    observations = env.reset()
    max_agents = env.num_agents
    total_rewards = np.zeros((env.num_agents, num_episodes))

    for episode in range(num_episodes):
        observations = env.reset()
        max_agents = env.num_agents
        agent_nums = {}
        for i, agent_name in enumerate(env.agents):
            agent_nums[agent_name] = i
        while env.agents:
            actions = {}
            for agent_name in env.agents:
                if agent_name[0:3] == "red":
                    actions[agent_name] = agent1.take_action(
                        observations[agent_name], id=agent_nums[agent_name]
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
                total_rewards[agent_nums[agent_name], episode] += rewards[agent_name]
            observations = new_observations
    return (
        np.mean(total_rewards[0 : max_agents // 2]),
        np.std(total_rewards[0 : max_agents // 2]),
        np.mean(total_rewards[max_agents // 2 :]),
        np.std(total_rewards[max_agents // 2 :]),
    )


if __name__ == "__main__":
    from CompetitionAgents.marlAgent.rand_agent import Random_Agent
    from CompetitionAgents.marlAgent.Q_net_example.Q_agent import Q_Agent

    agents = [Random_Agent(), Q_Agent()]
    comp_agent_folders = [
        "./CompetitionAgents/marlAgent/",
        "./CompetitionAgents/marlAgent/Q_net_example/",
    ]

    env = battle_v4.parallel_env(map_size=16, render_mode="human")

    for comp_agent in range(len(agents)):
        agents[comp_agent].load(comp_agent_folders[comp_agent])
        print("Loaded agent: ", comp_agent)
    for agent1 in agents:
        for agent2 in agents:
            mean1, std1, mean2, std2 = evaluate_agent(
                agent1, agent2, env, num_episodes=10
            )
            print(
                f"Agent 1: {agent1.__class__.__name__}, Agent 2: {agent2.__class__.__name__}"
            )
            print(f"Mean 1: {mean1}, Std 1: {std1},  Mean 2: {mean2}, Std 2: {std2}")
