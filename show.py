import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor):
        return self.actor.forward(x), self.critic.forward(x)

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

game = "LunarLander-v3"
# game = "CartPole-v1"
episodes = 5

env = gym.make(game, render_mode="human")

actor_critic = torch.load(f"{game}Net", weights_only=False)

for episode in range(episodes):
    state, _ = env.reset()

    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = actor_critic.actor.forward(state_tensor)
        action = torch.argmax(probs)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        state = next_state

