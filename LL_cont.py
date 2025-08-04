import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt

class ActorCriticContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable log_std
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        shared = self.shared(state)
        mean = self.mean_head(shared)
        value = self.value_head(shared)
        std = torch.exp(self.log_std)
        return mean, std, value

    def get_action_and_value(self, state):
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        squashed_action = torch.tanh(raw_action)

        # Rescale from [-1, 1] to actual action space
        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(-1)  # Tanh correction
        entropy = dist.entropy().sum(-1)
        return squashed_action, log_prob, entropy, value




reward = -100 * sqrt(x^2 + y^2)             # Distance to pad center
         -100 * sqrt(vel_x^2 + vel_y^2)     # Velocity penalty
         -100 * abs(angle)                  # Angle deviation penalty
         + 10 * leg1_contact                # Bonus for left leg contact
         + 10 * leg2_contact                # Bonus for right leg contact