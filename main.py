import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical



class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.fc(x)


def train_cartpole(env_name='CartPole-v1', episodes=1000, gamma=0.99, lr=1e-3):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr)

    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        values = []

        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy_net(state_tensor)
            value = value_net(state_tensor)

            m = Categorical(probs)
            action = m.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(m.log_prob(action))
            rewards.append(reward)
            values.append(value)

            state = next_state

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        values = torch.cat(values).squeeze()

        # Compute advantages
        advantages = returns - values.detach()

        # Update policy
        policy_loss = -(torch.stack(log_probs) * advantages).mean()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Update value network
        value_loss = nn.MSELoss()(values, returns)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        if (episode+1) % 50 == 0:
            print(f"Episode {episode+1}, Total Reward: {sum(rewards)}")

    return policy_net, value_net

def train_cartpole_td(env_name='CartPole-v1', episodes=1000, gamma=0.99, lr=1e-3):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr)

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Policy and value predictions
            probs = policy_net(state_tensor)
            value = value_net(state_tensor)

            # Sample action
            m = Categorical(probs)
            action = m.sample()

            # Step in env
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

            # Compute TD target
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            next_value = value_net(next_state_tensor) if not done else torch.tensor([[0.0]])
            td_target = reward + gamma * next_value.detach()
            advantage = td_target - value

            # Policy update (actor)
            policy_loss = -(m.log_prob(action) * advantage.detach())
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            # Value update (critic)
            value_loss = nn.MSELoss()(value, td_target)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            state = next_state

        if (episode+1) % 50 == 0:
            print(f"Episode {episode+1}, Total Reward: {total_reward}")

    return policy_net, value_net

def compute_gae(rewards, values, next_values, dones, gamma, lam):
    """
    rewards: [T] tensor
    values: [T] tensor of V(s_t)
    next_values: [T] tensor of V(s_{t+1})
    dones: [T] tensor (1 if done else 0)
    """
    deltas = rewards + gamma * next_values * (1 - dones) - values
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages

def train_cartpole_gae(env_name='CartPole-v1', episodes=4000, gamma=0.99, lam=0.95, lr=1e-3):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr)

    for episode in range(episodes):
        state, _ = env.reset()
        
        log_probs = []
        rewards = []
        values = []
        dones = []
        next_values = []

        done = False
        total_reward = []

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy_net(state_tensor)
            value = value_net(state_tensor)

            m = Categorical(probs)
            action = m.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            next_value = value_net(next_state_tensor) if not done else torch.tensor([[0.0]])

            log_probs.append(m.log_prob(action))
            rewards.append(torch.tensor([reward], dtype=torch.float32))
            values.append(value.squeeze())
            next_values.append(next_value.squeeze())
            dones.append(torch.tensor([float(done)]))

            state = next_state

        # Convert to tensors
        rewards = torch.cat(rewards)
        values = torch.stack(values)
        next_values = torch.stack(next_values)
        dones = torch.cat(dones)

        # Compute advantages using GAE
        advantages = compute_gae(rewards, values, next_values, dones, gamma, lam)
        returns = advantages + values  # target for critic

        # Policy loss
        log_probs = torch.stack(log_probs)
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = nn.MSELoss()(values, returns.detach())

        # Update policy
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Update value
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        if (episode+1) % 50 == 0:
            print(f"Episode {episode+1}, Total Reward: {total_reward}")

    return policy_net, value_net

def train_cartpole_ppo(env_name='CartPole-v1', episodes=2500, gamma=0.99, lam=0.95, actor_lr=1e-4, critic_lr=1e-3, eps_clip=0.2, batch_size = 128, ppo_batches = 4):
    def compute_gae(rewards, values, next_values, dones, gamma=gamma, lam=lam):
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
            # delta = rewards[t] + gamma * (next_values[t+1] if t < T - 1 else 0) * mask - next_values[t]
            delta = rewards[t] + gamma * (next_values[t]) * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae

        return advantages

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=actor_lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=critic_lr)
    value_loss_fn = nn.MSELoss()

    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        
        states = []
        actions = []
        rewards = []
        values = []
        dones = []
        old_log_probs = []
        next_values = []

        done = False

        while not done:
            states.append(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy_net(state_tensor)
            value = value_net(state_tensor)

            m = Categorical(probs)
            action = m.sample()
            actions.append(action)
            old_log_probs.append(m.log_prob(action).detach())

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            next_value = value_net(torch.tensor(next_state))
            next_values.append(next_value.detach())
            rewards.append(reward)
            values.append(value.squeeze())
            dones.append(float(done))

            state = next_state

        total_rewards.append(sum(rewards))

        for _ in range(ppo_batches):
            for j in range(0, len(rewards), batch_size):
                batch_start, batch_end = (j, j + batch_size)
                batch_states = torch.tensor(states[batch_start:batch_end])          # [batch, state_dim]
                batch_actions = torch.tensor(actions[batch_start:batch_end])
                batch_values = torch.tensor(values[batch_start:batch_end]).detach()
                batch_next_values = torch.tensor(next_values[batch_start:batch_end])
                batch_rewards = torch.tensor(rewards[batch_start:batch_end])
                batch_dones = torch.tensor(dones[batch_start:batch_end])
                # batch_old_probs = torch.stack(old_log_probs[batch_start:batch_end])
                batch_old_probs = torch.tensor(old_log_probs[batch_start:batch_end])

                # Compute advantages using GAE
                advantages = compute_gae(batch_rewards, batch_values, batch_next_values, batch_dones, gamma, lam)
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalize
                returns = advantages + batch_values  # critic target

                # PPO update
                m = Categorical(policy_net.forward(batch_states))
                batch_log_probs_new = m.log_prob(batch_actions)

                ratios = torch.exp(batch_log_probs_new - batch_old_probs)
                surr1 = ratios * advantages.detach()
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages.detach()
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values_pred = value_net(batch_states).squeeze()
                
                detached_returns = returns.detach()
                value_loss = 0.5 * value_loss_fn.forward(values_pred, detached_returns)

                # Update policy
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # Update value
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

        if (episode+1) % 50 == 0:
            print(f"Episode {episode+1}, Total Reward: {np.mean(total_rewards)}")
            total_rewards = []

    return policy_net, value_net

policy_net, value_net = train_cartpole_ppo()

env = gym.make("CartPole-v1", render_mode="human")

state, _ = env.reset()

done = False
while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    probs = policy_net(state_tensor)

    m = Categorical(probs)
    action = m.sample()

    next_state, reward, terminated, truncated, _ = env.step(action.item())
    done = terminated or truncated

    state = next_state

