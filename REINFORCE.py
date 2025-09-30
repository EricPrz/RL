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
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
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
        self.lengths = []

    def add(self, state, action, prob, reward, next_state, done) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def len(self) -> int:
        return self.states.__len__()

    def note_len(self, length):
        self.lengths.append(length)

    def get_batch(self, batch_size: int):
        pass

    def get_all(self):
        return self.states, self.actions, self.probs, self.rewards, self.next_states, self.dones, self.lengths

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.probs.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.lengths.clear()

# Function that compute returns from rewards
def compute_returns(rewards, gamma = 0.99):
    G = 0
    returns = torch.zeros(len(rewards))

    for T, reward in enumerate(reversed(rewards)):
        G = G * gamma + reward
        returns[-T-1] = G

    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns

class REINFORCE:
    def __init__(self, hidden_dim = 128, env_name="CartPole-v1", actor_lr=1e-4, critic_lr=1e-4):
        self.replay_buffer = ReplayBuffer()
        self.env = gym.make(env_name)
        self.model = ActorCritic(self.env.observation_space.shape[0], self.env.action_space.n, hidden_dim)
        self.actor_optim = torch.optim.Adam(self.model.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.model.critic.parameters(), lr=critic_lr)
        self.critic_loss = torch.nn.MSELoss()

    def train(self, episode_number = 10, epsilon = 0.2):
        env = self.env

        for episode in range(episode_number):
            state, _ = env.reset()

            done = False

            length = 0
            while not done:
                probs = self.model.actor(torch.tensor(state))
                m = Categorical(probs)
                action = m.sample()
                prob = m.log_prob(action)

                length += 1

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                self.replay_buffer.add(state, action.item(), prob, reward, next_state, done)

                state = next_state

            self.replay_buffer.note_len(length)


        if self.replay_buffer.len() <= 0:
            return
        states, actions, old_probs, rewards, next_states, dones, lengths = self.replay_buffer.get_all()

        for l in range(len(lengths)):
            
            batch_start = sum(lengths[:l])
            batch_end = batch_start + lengths[l]

            batch_rewards = rewards[batch_start:batch_end]
            batch_states = states[batch_start:batch_end]
            batch_actions = actions[batch_start:batch_end]
            batch_old_probs = self.model.actor(torch.tensor(batch_states))
            batch_old_probs = Categorical(batch_old_probs).log_prob(torch.tensor(batch_actions))

            returns = compute_returns(batch_rewards)

            V = self.model.critic(torch.tensor(batch_states)).squeeze()

            advantage = returns - V.detach()


            self.actor_optim.zero_grad()
            actor_loss = -(batch_old_probs * advantage).mean()
            actor_loss.backward()
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss = 0.5 * self.critic_loss(V, returns)
            critic_loss.backward()
            self.critic_optim.step()


            

        self.replay_buffer.clear()

    def eval(self, episode_number = 15):
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
        return total_rewards/episode_number


reinforce = REINFORCE(actor_lr=3e-5, critic_lr=1e-4)
for i in range(20):
    reinforce.model.train()
    reinforce.train(100)
    reinforce.model.eval()
    rew = reinforce.eval()
    if rew >= 500:
        env = gym.make("CartPole-v1", render_mode = "human")

        state, _ = env.reset()
        done = False

        while not done:
            probs = reinforce.model.actor(torch.tensor(state))
            action = Categorical(probs).sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            done = terminated or truncated
            state = next_state

        break



