from CompetitionAgents.singleAgent.Q_net_example.Q_agent import Q_Agent
from CompetitionAgents.marlAgent.rand_agent import Random_Agent
from CompetitionAgents.singleAgent.Q_net_example.Q_agent import Mem_Buffer
from Agent import Agent
import gymnasium as gym
import torch
import pygame


def train_loop(
    num_episodes,
    agent: Agent,
    env,
    render=False,
    comp_agent_folders=None,
):
    observation = env.reset()
    mem_buffer = Mem_Buffer(max_size=10000, obs_dim=16 * 16 * 3, action_dim=5)
    episode_reward_history = []

    epsilon = 1.0  # exploration rate
    epsilon_decay = 0.98  # decay rate for exploration probability

    for episode in range(num_episodes):
        keys = {"w": False, "a": False, "s": False, "d": False}
        epsilon = max(0.05, epsilon * epsilon_decay)  # decay the exploration rate
        print(epsilon)
        tot_reward = 0
        observation, info = env.reset()
        observation = agent.transform_obs(observation)
        step = 0
        terminated, truncated = False, False
        while not (terminated or truncated):

            action = -1
            for event in pygame.event.get():
                # get wasd input
                if event.type == pygame.KEYDOWN:
                    print("kes")
                    if event.key == pygame.K_w:
                        keys["w"] = True
                    elif event.key == pygame.K_a:
                        keys["a"] = True
                    elif event.key == pygame.K_s:
                        keys["s"] = True
                    elif event.key == pygame.K_d:
                        keys["d"] = True
                    elif event.key == pygame.K_ESCAPE:
                        exit(0)
                    print(action)
                if event.type == pygame.KEYUP:
                    print("kes")
                    if event.key == pygame.K_w:
                        keys["w"] = False
                    elif event.key == pygame.K_a:
                        keys["a"] = False
                    elif event.key == pygame.K_s:
                        keys["s"] = False
                    elif event.key == pygame.K_d:
                        keys["d"] = False
                    elif event.key == pygame.K_ESCAPE:
                        exit(0)
                    print(keys)
            if keys["w"]:
                action = 3
            if keys["a"]:
                action = 2
            if keys["s"]:
                action = 4
            if keys["d"]:
                action = 1
            if action == -1:
                action = agent.take_action(observation, id=0)
            new_observation, reward, terminated, truncated, info = env.step(action)
            new_observation = agent.transform_obs(new_observation)
            tot_reward += reward

            mem_buffer.add(
                observation.flatten(),
                action,
                reward,
                new_observation.flatten(),
                float(terminated),
            )

            # every 16 steps we train the agents
            if step % 16 == 0:
                if len(mem_buffer) > 128:
                    states, actions, rewards, states_, dones = mem_buffer.sample(128)
                    states = torch.tensor(states).float().to(agent.device)
                    actions = torch.tensor(actions).to(agent.device)
                    rewards = torch.tensor(rewards).float().to(agent.device)
                    states_ = torch.tensor(states_).float().to(agent.device)
                    dones = torch.tensor(dones).to(agent.device)

                    q_values = agent.Q_net(
                        states
                    )  # get q values for each action for current states
                    q_values = torch.gather(
                        q_values, dim=-1, index=actions.long()
                    ).squeeze(
                        -1
                    )  # Get the q values for the action we chose
                    with torch.no_grad():
                        next_q_values = torch.max(agent.Q_net(states_), dim=-1).values
                        targets = rewards + 0.99 * (1 - dones) * next_q_values

                    loss = ((q_values - targets) ** 2).mean()  # calculate loss
                    agent.optimizer.zero_grad()
                    loss.backward()  # backpropagate loss
                    agent.optimizer.step()  # update weights

            observation = new_observation.copy()

        step += 1
        print(tot_reward)
        episode_reward_history.append(tot_reward)
        if comp_agent_folders is not None:
            agent.save(comp_agent_folders[0])
    return episode_reward_history


if __name__ == "__main__":
    agents = [Q_Agent(obs_dim=16 * 16 * 3, action_dim=5, hidden_dim=64), Random_Agent()]
    comp_agent_folders = [
        "./CompetitionAgents/marlAgent/Q_net_example/",  # Q agent
    ]

    agents[0].save(comp_agent_folders[0])  # save the agent to the folder
    agents[0].load(comp_agent_folders[0])

    env = gym.make(
        "CarRacing-v2",  # v3 for linux
        render_mode="human",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=False,
    )
    train_loop(
        100,
        agents[0],
        env,
        render=False,
        comp_agent_folders=comp_agent_folders,
    )
