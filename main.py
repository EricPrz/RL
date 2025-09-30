import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from gymnasium.wrappers import NormalizeObservation
from gymnasium.wrappers import NormalizeReward

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.norm = nn.BatchNorm1d(state_dim)
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



class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
            nn.Linear(hidden_dim, hidden_dim),
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

def train_cartpole_ppo(env_name='CartPole-v1', load = False, solved_reward=500.0, max_episodes=2500, eval_episodes = 10, gamma=0.99, lam=0.95, actor_lr=1e-4, critic_lr=1e-3, eps_clip=0.2, batch_size = 128, ppo_batches = 4):
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
            delta = rewards[t] + gamma * next_values[t].detach() * mask - values[t].detach()
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae

        return advantages

    env = gym.make(env_name)
    # env = NormalizeReward(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # policy_net = PolicyNetwork(state_dim, action_dim)
    # value_net = ValueNetwork(state_dim)
    if load:
        actor_critic = torch.load(f"{game}Net", weights_only=False)
    else:
        actor_critic = ActorCritic(state_dim, action_dim)
    critic_optim = optim.Adam(actor_critic.critic.parameters(), lr = critic_lr)
    actor_optim = optim.Adam(actor_critic.actor.parameters(), lr = actor_lr)

    critic_loss_fn = nn.MSELoss()

    # policy_optimizer = optim.Adam(policy_net.parameters(), lr=actor_lr)
    # value_optimizer = optim.Adam(value_net.parameters(), lr=critic_lr)
    # value_loss_fn = nn.MSELoss()

    total_rewards = []
    episode = 0
    max_reward = 0

    actor_critic.train()
    while episode < max_episodes and max_reward < solved_reward:
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
            probs = actor_critic.actor.forward(state_tensor)
            value = actor_critic.critic(state_tensor)

            m = Categorical(probs)
            action = m.sample()
            actions.append(action)
            old_log_probs.append(m.log_prob(action))

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            next_value = actor_critic.critic.forward(torch.tensor(next_state))
            next_values.append(next_value)
            rewards.append(reward)
            values.append(value.squeeze())
            dones.append(float(done))

            state = next_state

        total_rewards.append(sum(rewards))


        advantages = compute_gae(rewards, values, next_values, dones, gamma, lam)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalize

        states = torch.tensor(states)          # [batch, state_dim]
        actions = torch.tensor(actions)
        values = torch.stack(values)
        old_probs = torch.stack(old_log_probs)

        for _ in range(ppo_batches):
            for j in range(0, len(rewards), batch_size):
                batch_start, batch_end = (j, j + batch_size)
                batch_end = min(batch_start + batch_size, len(rewards))

                batch_states = states[batch_start:batch_end]
                batch_actions = actions[batch_start:batch_end]
                batch_values = values[batch_start:batch_end]
                batch_advantages = advantages[batch_start:batch_end]
                batch_old_probs = old_probs[batch_start:batch_end]


                # PPO update
                m = Categorical(actor_critic.actor.forward(batch_states))
                batch_log_probs_new = m.log_prob(batch_actions)

                ratios = torch.exp(batch_log_probs_new.detach() - batch_old_probs.detach())
                surr1 = ratios * batch_advantages.detach()
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * batch_advantages.detach()
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values_pred = actor_critic.critic.forward(batch_states).view(-1)
                
                # Compute advantages using GAE
                returns = batch_advantages + batch_values.detach()  # critic target
                detached_returns = returns.detach()

                value_loss = 0.5 * critic_loss_fn.forward(values_pred, detached_returns)

                # Update policy
                actor_optim.zero_grad()
                policy_loss.backward()
                actor_optim.step()

                # Update value
                critic_optim.zero_grad()
                value_loss.backward()
                critic_optim.step()

        if (episode+1) % 50 == 0:
            print(f"Episode {episode+1}, Total Reward: {np.mean(total_rewards)}")
            max_reward = np.mean(total_rewards)
            total_rewards = []

        episode += 1

    return actor_critic


def train_cartpole_ppo_chat(env_name='CartPole-v1', load = False, solved_reward=500.0, max_episodes=2500, eval_episodes = 10, gamma=0.99, lam=0.95, actor_lr=1e-4, critic_lr=1e-3, eps_clip=0.2, batch_size = 128, ppo_batches = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_gae(rewards, values, next_values, dones, gamma=gamma, lam=lam):
        """
        rewards: list or 1D tensor length T
        values: tensor shape [T] (V(s_t))
        next_values: tensor shape [T] (V(s_{t+1}) for each step; for final step this can be 0 or bootstrap)
        dones: list or tensor length T (1.0 if done after step t, else 0.0)
        returns advantages tensor shape [T]
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=device, dtype=torch.float32)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])  # 0 if done, 1 otherwise
            # use next_values[t] as V(s_{t+1})
            delta = float(rewards[t]) + gamma * (float(next_values[t]) * mask) - float(values[t])
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
        return advantages

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # assume ActorCritic is your defined model class; ensure it .to(device)
    if load:
        actor_critic = torch.load(f"{env_name}Net", map_location=device)
    else:
        actor_critic = ActorCritic(state_dim, action_dim)
    actor_critic.to(device)

    critic_optim = optim.Adam(actor_critic.critic.parameters(), lr = critic_lr)
    actor_optim = optim.Adam(actor_critic.actor.parameters(), lr = actor_lr)
    critic_loss_fn = nn.MSELoss()

    total_rewards = []
    episode = 0
    max_reward = -np.inf

    actor_critic.train()
    while episode < max_episodes and max_reward < solved_reward:
        state, _ = env.reset()
        done = False

        # rollout buffers for one episode
        states = []
        actions = []
        rewards = []
        values = []
        dones = []
        old_log_probs = []
        next_values = []

        while not done:
            states.append(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # [1, state_dim]

            with torch.no_grad():
                probs = actor_critic.actor(state_tensor)  # assume gives probabilities
                value = actor_critic.critic(state_tensor).view(-1)  # shape [1]

            m = Categorical(probs)
            action = m.sample()
            logp = m.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            step_done = terminated or truncated

            # store
            actions.append(action)
            old_log_probs.append(logp.detach())    # detach old log prob (stale policy)
            values.append(value.squeeze().detach()) # store V(s_t) as scalar (we'll use for returns)
            rewards.append(reward)
            dones.append(float(step_done))

            # compute V(s_{t+1}) for bootstrap; note: use spectator critic without grad
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                next_v = actor_critic.critic(next_state_tensor).view(-1)
            next_values.append(next_v.squeeze().detach())

            state = next_state
            done = step_done

        total_rewards.append(sum(rewards))

        # Convert lists to tensors on device
        states = torch.FloatTensor(states).to(device)        # [T, state_dim]
        actions = torch.stack(actions).to(device)            # [T]
        values = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in values]).to(device).float()  # [T]
        next_values = torch.stack([nv if isinstance(nv, torch.Tensor) else torch.tensor(nv) for nv in next_values]).to(device).float()  # [T]
        old_log_probs = torch.stack(old_log_probs).to(device).float()  # [T]
        rewards_t = rewards  # python list
        dones_t = dones      # python list

        # compute advantages (tensor)
        advantages = compute_gae(rewards_t, values, next_values, dones_t, gamma, lam)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = advantages + values  # shape [T]

        # PPO updates: multiple epochs over the collected trajectory
        T = len(rewards)
        for _ in range(ppo_batches):
            # shuffle minibatches
            perm = torch.randperm(T)
            for start in range(0, T, batch_size):
                idx = perm[start:start+batch_size]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_values = values[idx]
                batch_adv = advantages[idx]
                batch_old_logp = old_log_probs[idx]
                batch_returns = returns[idx].detach()

                # get new log probs and value preds
                probs_new = actor_critic.actor(batch_states)
                m_new = Categorical(probs_new)
                batch_logp_new = m_new.log_prob(batch_actions)  # important: not detached

                # ratio for policy update (new_logp - old_logp)
                ratios = torch.exp(batch_logp_new - batch_old_logp)

                surr1 = ratios * batch_adv
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss (MSE to returns)
                values_pred = actor_critic.critic(batch_states).view(-1)
                value_loss = 0.5 * critic_loss_fn(values_pred, batch_returns)

                # update actor
                actor_optim.zero_grad()
                policy_loss.backward()
                actor_optim.step()

                # update critic
                critic_optim.zero_grad()
                value_loss.backward()
                critic_optim.step()

        # logging
        if (episode+1) % 10 == 0:
            mean_recent = np.mean(total_rewards[-50:]) if len(total_rewards) >= 1 else np.mean(total_rewards)
            print(f"Episode {episode+1}, Episode Reward: {total_rewards[-1]:.2f}, Mean(recent): {mean_recent:.2f}")
            max_reward = mean_recent

        episode += 1

    return actor_critic

game = "LunarLander-v3"
game = "CartPole-v1"

actor_crit = train_cartpole_ppo(env_name=game, load = False, solved_reward=240.0, max_episodes=2000, actor_lr=1e-4, critic_lr=1e-3, gamma=0.9, batch_size=128)


torch.save(actor_crit, f"{game}Net")
print(f"Model {game}Net saved")

env = gym.make(game, render_mode="human")
state, _ = env.reset()

actor_crit.eval()
done = False
while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    probs = actor_crit.actor.forward(state_tensor)
    action = torch.argmax(probs)

    next_state, reward, terminated, truncated, _ = env.step(action.item())
    done = terminated or truncated

    state = next_state

