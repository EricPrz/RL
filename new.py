import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from torch.nn.modules import loss

# ActorCritic NN's
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(state_dim)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        return self.actor(x), self.critic(x)

# Class for managing replay buffers
class ReplayBuffer:
    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, prob, reward, next_state, done) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def len(self) -> int:
        return self.states.__len__()

    def get_batch(self, batch_size: int):
        pass

    def get_all(self):
        return self.states, self.actions, self.probs, self.rewards, self.next_states, self.dones

# Function that compute returns from rewards
def compute_returns(rewards, gamma = 0.9):
    G = 0
    returns = torch.zeros(len(rewards))

    for T, reward in enumerate(reversed(rewards)):
        G = G * gamma + reward
        returns[T] = G

    return returns.__reversed__()

def compute_gae(rewards, values, next_values, dones, gamma=0.9, lam=0.95):
        """
        rewards: list or 1D tensor length T
        values: tensor shape [T] (V(s_t))
        next_values: tensor shape [T] (V(s_{t+1}) for each step; for final step this can be 0 or bootstrap)
        dones: list or tensor length T (1.0 if done after step t, else 0.0)
        returns advantages tensor shape [T]
        """
        T = len(rewards)
        advantages = torch.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])  # 0 if done, 1 otherwise
            # use next_values[t] as V(s_{t+1})
            delta = rewards[t] + gamma * (next_values[t] * mask) - (values[t])
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
        return advantages

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Tensor of shape [T] (float)
            values: Tensor of shape [T] (float)
            next_values: Tensor of shape [T] (float)
            dones: Tensor of shape [T] (0.0 or 1.0)
            gamma: discount factor
            lam: GAE lambda

        Returns:
            advantages: Tensor of shape [T]
        """
        T = len(rewards)
        advantages = torch.zeros(T)
        gae = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]           # 0 if done, 1 otherwise
            delta = rewards[t] + gamma * (values[t+1] if t < T - 1 else 0) * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae

        return advantages

class PPO:
    def __init__(self, hidden_dim = 128, env_name="CartPole-v1"):
        self.replay_buffer = ReplayBuffer()
        self.env = gym.make(env_name)
        self.model = ActorCritic(self.env.observation_space.shape[0], self.env.action_space.n, hidden_dim)
        self.actor_optim = torch.optim.Adam(self.model.actor.parameters(), lr=1e-4)
        self.critic_optim = torch.optim.Adam(self.model.critic.parameters(), lr=1e-3)
        self.critic_loss_fn = loss.MSELoss()

    def train(self, episode_number = 10, epsilon = 0.2):
        env = self.env

        self.replay_buffer = ReplayBuffer()
        for episode in range(episode_number):
            state, _ = env.reset()

            done = False

            while not done:
                probs = self.model.actor(torch.tensor(state))
                m = Categorical(probs)
                action = m.sample()
                prob = m.log_prob(action)

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                self.replay_buffer.add(state, action.item(), prob.detach(), reward, next_state, done)

                state = next_state


            if self.replay_buffer.len() <= 0:
                continue
            states, actions, old_probs, rewards, next_states, dones = self.replay_buffer.get_all()

            probs, V = self.model(torch.tensor(states))
            prob = Categorical(probs).log_prob(torch.tensor(actions))
            V = V.detach()
            old_probs = torch.tensor(old_probs)

            advantage = compute_gae(rewards, V, dones)
            if len(advantage) > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # normalize


            ratio = torch.exp(prob - old_probs)
            surr1 = ratio * advantage.detach()
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage.detach()
            loss = -torch.min(surr1, surr2).mean()


            _, V = self.model(torch.tensor(states))
            returns = advantage + V.squeeze()
            critic_loss = 0.5 * self.critic_loss_fn(V.squeeze(), returns.detach())

            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

    def eval(self, episode_number = 5):
        env = self.env

        total_rewards = 0
        for episode in range(episode_number):
            state, _ = env.reset()
            done = False
            total_reward = 0 

            while not done:
                probs = self.model.actor(torch.tensor(state))
                action = Categorical(probs).sample()

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                total_reward += reward

                done = terminated or truncated
                state = next_state


            total_rewards += total_reward
        print("Rewards:", total_rewards/episode_number)


ppo = PPO()
for i in range(20):
    ppo.train(100)
    ppo.eval()



