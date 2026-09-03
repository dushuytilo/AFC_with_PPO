"""
This file contains the actor and critic network for the PPO.
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli


class ActorNetwork(nn.Module):
    """Policy Model"""

    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(ActorNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )

    def forward(self, state):
        """Runs a forward pass on the NN"""
        logit = self.actor(state)

        prob = torch.sigmoid(logit)
        dist = Bernoulli(prob)  # p = P(True); 1-p = P(False) #action space is binary

        return dist


class CriticNetwork(nn.Module):
    """Value Model"""

    def __init__(self, num_inputs, hidden_size):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        """Runs a forward pass on the NN"""
        value = self.critic(state)
        return value
