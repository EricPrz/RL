import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(observation_space, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, action_space)
        )

    def forward(self, x):
        x = torch.tensor(x)
        out = self.seq.forward(x)
        return out


env = gym.make("CartPole-v1")

model = Model(4, 2)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

episodes = 1000

points = []

for j in range(episodes):

    observation, info = env.reset()
    ended = False

    rewards = []
    probs = []

    while (not ended):

        logits = model(observation)
        m = torch.distributions.Categorical(logits=logits)
        action = m.sample()

        observation, reward, terminated, truncated, info = env.step(action.item())

        rewards.append(reward)
        probs.append(m.log_prob(action))

        if (terminated or truncated):
            ended = True


    # Computing Returns
    G = 0
    returns = np.zeros(len(rewards))
    gamma = 0.99
    for i in reversed(range(len(rewards))):
        G = G * gamma + rewards[i]
        returns[i] = G

    # Updating Model
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = -(torch.stack(probs) * returns).sum()

    loss.backward()
    if (j % 10 == 0):
        optimizer.step()
        optimizer.zero_grad()

    total_reward = 0
    for i in range(len(rewards)):
        total_reward += rewards[i]

    points.append(total_reward)

plt.plot(points)
plt.show()
