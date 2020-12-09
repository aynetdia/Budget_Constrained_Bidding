"""
Adapted from: https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/model.py

The code was modified to add one more hidden layer as suggested by the paper: 
Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising
(https://arxiv.org/pdf/1802.08365.pdf)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random


class Network(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=100, 
                    fc2_units=100, fc3_units=100):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Network, self).__init__()
        set_seed()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

def set_seed():
  os.environ['PYTHONHASHSEED'] = str(0)
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed_all(0)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
