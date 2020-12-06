import sys,os
sys.path.append(os.getcwd()+'/src/gym-auction_emulator')
import gym, gym_auction_emulator
import random
import torch
import numpy as np
from collections import deque

import configparser
from dqn import Agent
from reward_net import RewardNet
import numpy as np


class RlBidAgent():

    def _load_config(self):
        """
        Parse the config.cfg file
        """
        cfg = configparser.ConfigParser(allow_no_value=True)
        env_dir = os.path.dirname(__file__)
        cfg.read(env_dir + '/config.cfg')
        self.budget = int(cfg['agent']['budget'])
        self.T = int(cfg['rl_agent']['T']) # T number of timesteps
        self.STATE_SIZE = int(cfg['rl_agent']['STATE_SIZE'])
        self.ACTION_SIZE = int(cfg['rl_agent']['ACTION_SIZE'])
        
    
    def __init__(self):
        self._load_config()
        # Control parameter used to scale bid price
        self.BETA = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]
        self.eps = 0.9
        self.anneal = 1e-5
        self.t_step = 0
        self.budget_spend = 0.0
        self.reward_t = 0
        self.ROL = self.T # 3. the number of Lambda regulation opportunities left
        self._reset_episode()
        # DQN Network to learn Q function
        self.dqn_agent = Agent(state_size = self.STATE_SIZE, action_size=self.ACTION_SIZE, seed =0)
        # Reward Network to reward function
        self.reward_net = RewardNet(state_action_size = self.STATE_SIZE + 1, reward_size=1, seed =0)
        self.dqn_state = None
        self.dqn_action = 3 # first action - no adjustment
        self.total_wins = 0
        self.total_rewards = 0.0
        self.total_cost = 0

    def _reset_episode(self):
        """
        Function to reset the state when episode changes
        """
        self._reset_step()
        self.cur_day = 0
        self.cur_hour = 0
        self.ctl_lambda = 1e-3  # Lambda sequential regulation parameter
        self.wins_e = 0  
        self.reward_net.V = 0
        self.reward_net.S = []

    def _update_step(self):
        """
        Function that is called before transitioning into step t+1 (updates state t)
        """
        self.t_step += 1
        self.prev_budget = self.rem_budget # Bt-1
        self.rem_budget = self.budget - self.budget_spend # Bt (2. the remaining budget at time-step t)
        self.ROL -= 1
        self.BCR = (self.rem_budget - self.prev_budget) / self.prev_budget
        self.CPM = self.cost_t / (self.wins_t/1000) # cost of the won ads divided by the number of impressions (expressed in thousands)
        self.WR = self.total_wins / self.imp_opps 

    def _reset_step(self):
        """
        Function to call every time a new time step is entered.
        """
        self.rewards_prev_t = self.reward_t # 7. total clicks at timestep t-1
        self.reward_t = 0.
        self.cost_t = 0.
        self.wins_t = 0
        self.bids_t = 0
        self.imp_opps = 0
        self.BCR = 0                      # 4. Budget consumption rate
        self.CPM = 0                      # 5. Cost per mille of impressions between t-1 and t:
        self.WR = 0                       # 6. total wins / total_impression opportunities
        self.eps = max(0.95 - self.anneal * self.t_step, 0.05)
    
    def _update_reward_cost(self, reward, cost):
        """
        Internal function to update reward and action to compute the cumulative
        reward and cost within the given step.
        """
        self.wins_t += 1
        self.wins_e += 1
        self.total_wins += 1
        self.reward_t += reward # click
        self.total_rewards += reward
        self.cost_t += cost
        self.total_cost += cost
        
    def _get_state(self):
        """
        Returns the state that will be used for the DQN state.
        """
        return np.asarray([self.t_step,
                self.rem_budget,
                self.ROL,
                self.BCR,
                self.CPM,
                self.WR,
                self.rewards_prev_t])

    def act(self, state, reward, cost):
        """
        This function gets called with every bid request.
        By looking at the weekday and hour to progress between the steps and
        episodes during training.
        Returns the bid request cost based on the scaled version of the
        bid price using the DQN agent output.
        """
        episode_done = (state['weekday'] != self.cur_day)
        # within the time step
        if state['hour'] == self.cur_hour and state['weekday'] == self.cur_day:
            pass
        # within the episode, changing the time step
        elif state['hour'] != self.cur_hour and state['weekday'] == self.cur_day:
            self._update_step()
            self.reward_net.step() # update reward net
            dqn_next_state = self._get_state() # observe state s_1 (state at the beginning of t=1)
            # get action a_1 (adjusting lambda_0 to lambda_1) from the adaptive greedy policy
            a_beta = self.dqn_agent.act(dqn_next_state, eps=self.eps)
            self.ctl_lambda *= (1 + self.BETA[a_beta]) # adjust lambda t-1 to t (0 to 1)
            sa = np.append(self.dqn_state, self.dqn_action) # state-action pair for t=0
            rnet_r = float(self.reward_net.act(sa)) # get reward r_0 from RewardNet
            self.reward_net.V += self.reward_t
            self.reward_net.S.append((self.dqn_state, self.dqn_action))
            # call agent step
            # Sample a mini batch and perform grad-descent step
            self.dqn_agent.step(self.dqn_state, self.dqn_action, rnet_r, dqn_next_state, episode_done)
            self.dqn_state = dqn_next_state # set state t+1 as state t (in order call it during next transition)
            self.dqn_action = a_beta # analogously with the action t+1
            self.cur_hour = state['hour'] 
            self._reset_step()
        # episode changes
        elif state['weekday'] != self.cur_day:
            for (s, a) in self.reward_net.S:
                sa = tuple(np.append(s, a))
                max_r = max(self.reward_net.get_from_M(sa), self.reward_net.V)
                self.reward_net.add_to_M(sa, max_r)
                self.reward_net.add(sa, max_r)
            print("Total Impressions won with Budget={} Spend={} wins = {}".format(self.budget, self.budget_spend, self.wins_e))
            self.total_wins += self.wins_e
            self._reset_episode() 
            self.cur_day = state['weekday']
            self.cur_hour = state['hour']

        self.imp_opps += 1
        bid = self.bid(state['click'], reward, cost) # bid with lambda_0

        return bid

    def bid(self, imp_value, reward, cost):
        bid_amt = imp_value/self.ctl_lambda
        if bid_amt > cost:
            self.budget_spend += cost
            self._update_reward_cost(reward, cost)
        if bid_amt > 0
            self.bids_t += 1
        return bid_amt

def main():
    # Instantiate the Environment and Agent
    env = gym.make('AuctionEmulator-v0')
    env.seed(0)
    agent = RlBidAgent()

    obs, reward, cost, done = env.reset()
    agent.cur_day = obs['weekday']
    agent.cur_hour = obs['hour']
    agent.dqn_state = agent._get_state() # observe state s_0

    while not done:
        # action = bid amount
        bid = agent.act(obs, reward, cost) # obs = state
        next_obs, next_reward, next_cost, done = env.step(bid)
        obs, reward, cost = next_obs, next_reward, next_cost # Next state assigned to current state
        if cost > (agent.budget - agent.total_cost):
            done == True

    print("Total Impressions won {} value = {}".format(agent.total_wins, agent.total_rewards))
    env.close()


if __name__ == "__main__":
    main()
