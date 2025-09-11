import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


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
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.0000002)

episodes = 3000

points = []
raw_points = []

batch_size = 10
gamma = 0.99

for j in range(int(episodes/batch_size)):
# for j in range(episodes):
    print(j)
    batch_rewards = []
    batch_log_probs = []

    for _ in range(batch_size):
        observation, info = env.reset()
        ended = False

        rewards = []
        log_probs = []

        while not ended:
            logits = model(observation)
            m = torch.distributions.Categorical(logits=logits)
            action = m.sample()

            observation, reward, terminated, truncated, info = env.step(action.item())

            rewards.append(reward)
            log_probs.append(m.log_prob(action))

            if terminated or truncated:
                ended = True

        raw_points.append(sum(rewards))
        # compute returns
        G = 0
        returns = np.zeros(len(rewards))
        for i in reversed(range(len(rewards))):
            G = G * gamma + rewards[i]
            returns[i] = G

        # normalize returns per episode
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        batch_rewards.append(sum(rewards))
        batch_log_probs.append((torch.stack(log_probs), returns))

    # ---- UPDATE STEP ----
    optimizer.zero_grad()

    # flatten all episodes into one loss
    all_losses = []
    for log_probs, returns in batch_log_probs:
        all_losses.append(-(log_probs * returns).sum())

    loss = torch.stack(all_losses).mean()  # normalize across episodes

    loss.backward()
    optimizer.step()

    # track mean reward for plotting
    points.append(np.mean(batch_rewards))

x = np.arange(0, episodes)
res = stats.linregress(x, raw_points)
plt.plot(raw_points, 'r', alpha=0.2)
plt.plot(np.arange(0, episodes, batch_size), points)
plt.plot(x, res.intercept + res.slope*x, 'g')
plt.ylabel("Reward")
plt.xlabel("Episode")
plt.show()
