# https://github.com/udacity/deep-reinforcement-learning/blob/master/solution/dqn_agent.py

# Modified batch size to 32
# gamma is set to 1

import numpy as np
import random
from collections import namedtuple, deque

from model import Network

import torch
import torch.nn as nn
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32         # minibatch size
GAMMA = 1.0             # discount factor
LR = 1e-3               # learning rate 
MOMENTUM = 0.95         # momentum
UPDATE_EVERY = 25       # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = Network(state_size, action_size, seed).to(device)
        self.qnetwork_target = Network(state_size, action_size, seed).to(device)
        self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=LR, momentum=MOMENTUM)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.loss = 0
    
    def step(self, state, action, reward, next_state, eval_flag):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state)
        self.t_step = state[0]
        
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > BATCH_SIZE and eval_flag == False:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        # Check if the Q-value distribution is unimodal or not
        #if np.count_nonzero(action_values == max_q) > 1:
        if unimodal_check(action_values) == True:
            return np.argmax(action_values.cpu().data.numpy())
        # If not unimodal, randomly choose an action with p = max(eps, 0.5)
        else: # if unimodal, choose an action regularly 
            prob = max(eps, 0.5)
            if random.random() <= prob:
                return random.choice(np.arange(self.action_size))
            else: # and with 1-p choose an action regularly 
                return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        if next_states[2] == 0:
            y = rewards
        else:
            y = rewards + gamma * self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # # Get max predicted Q values (for next states) from target model
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # # Compute Q targets for current states 
        # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        self.loss = nn.MSELoss(Q_expected, y)
        print("DQN loss = {}".format(self.loss))
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Every C steps reset Q target = Q (hard copy)
        if ((self.t_step + 1) % UPDATE_EVERY) == 0:
            for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
                target_param.data.copy_(local_param.data)

    def unimodal_check(self, action_values):
        """
        This function checks if the array of action-values is unimodal using
        some heuristic tests.
        :param action_values: predicted Q values for each action (sorted by default)
        :return: boolean variable describing whether the distribution of values
        in the action-value array is unimodal or "abnormal".
        """
        end = len(action_values)
        i = 1
        if max(action_values) == action_values[0] or max(action_values) == action_values[-1]:
            while i < end and action_values[i-1] > action_values[i]:
                i += 1
            while i < end and action_values[i-1] == action_values[i]:
                i += 1
            while i < end and action_values[i-1] < action_values[i]:
                i += 1
            return i == end
        else:
            while i < end and action_values[i-1] < action_values[i]:
                i += 1
            while i < end and action_values[i-1] == action_values[i]:
                i += 1
            while i < end and action_values[i-1] > action_values[i]:
                i += 1
            return i == end

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
  
        return (states, actions, rewards, next_states)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
