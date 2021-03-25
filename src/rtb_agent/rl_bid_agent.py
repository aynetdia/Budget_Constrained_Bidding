import sys,os
sys.path.append(os.getcwd()+'/src/gym-auction_emulator')
import gym, gym_auction_emulator
import random
from operator import itemgetter
import torch
import numpy as np
from collections import deque
import cloudpickle

import configparser
from dqn import DQN
from reward_net import RewardNet
from model import set_seed
import numpy as np
import pandas as pd


class RlBidAgent():

    def _load_config(self):
        """
        Parse the config.cfg file
        """
        cfg = configparser.ConfigParser(allow_no_value=True)
        env_dir = os.path.dirname(__file__)
        cfg.read(env_dir + '/config.cfg')
        self.exp_type = str(cfg['experiment_type']['type'])
        self.T = int(cfg[self.exp_type]['T']) # Number of timesteps in each episode
    
    def __init__(self):
        self._load_config()
        # Beta parameter adjsuting the lambda parameter, that regulates the agent's bid amount
        self.BETA = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]
        # Starting value of epsilon in the adaptive eps-greedy policy
        self.eps = 0.9
        # Parameter controlling the annealing speed of epsilon
        self.anneal = 2e-5
        if self.exp_type in ('improved_drlb', 'improved_drlb_eval'):
            # DQN Network to learn Q function
            self.dqn_agent = DQN(state_size = 6, action_size = 7)
            # Reward Network to learn the reward function
            self.reward_net = RewardNet(state_action_size = 7, reward_size = 1)
        else:
            self.dqn_agent = DQN(state_size = 7, action_size = 7)
            self.reward_net = RewardNet(state_action_size = 8, reward_size = 1)
        # Number of timesteps in each episode (4 15min intervals x 24 hours = 96)
        # self.T = 672
        # Initialize the DQN action for t=0 (index 3 - no adjustment of lambda, 0 ind self.BETA)
        self.dqn_action = 3
        self.ctl_lambda = None
        # Arrays saving the training history
        self.step_memory = []
        self.episode_memory = []
        # Params for tracking the progress
        self.global_T = 0 # Tracking the global time step
        self.episode_budgets = None
        self.budget = None
        self.total_wins = 0
        self.total_rewards = 0
        self.rewards_prev_t = 0
        self.rewards_prev_t_ratio = 0
        self.rnet_r = 0
        self.wins_e = 0
        self.rewards_e = 0
        self.ROL = self.T
        self.ROL_ratio = 1

    def _get_state(self):
        """
        Returns the state that will be used as input in the DQN
        """
        if self.exp_type in ('improved_drlb', 'improved_drlb_eval'):
            return np.asarray([self.rem_budget_ratio, # 2. the ratio of the remaining budget to total available budget at time-step t
                self.ROL_ratio, # 3. The ratio of the number of Lambda regulation opportunities left 
                self.BCR, # 4. Budget consumption rate
                self.CPI, # 5. Cost per impression between t-1 and t, in relation to the highest cost possible in the training set (300)
                self.WR, # 6. Auction win rate at state t
                self.rewards_prev_t_ratio]) # 7. Ratio of acquired/total clicks at timestep t-1
        else:
            return np.asarray([self.t_step, # 1. Current time step
                    self.rem_budget, # 2. the remaining budget at time-step t
                    self.ROL, # 3. The number of Lambda regulation opportunities left
                    self.BCR, # 4. Budget consumption rate
                    self.CPM, # 5. Cost per mille of impressions between t-1 and t:
                    self.WR, # 6. Auction win rate at state t
                    self.rewards_prev_t]) # 7. Clicks acquired at timestep t-1

    def _reset_episode(self):
        """
        Function to reset the state when episode changes
        """
        # Reset the count of time steps
        self.t_step = 0
        # Lambda regulation parameter - set according to the greedy approximation algorithm, as suggested by the paper
        if self.exp_type == 'vanilla_drlb':
            self.ctl_lambda = 0.01 if self.budget is None else self.calc_greedy(self.greedy_memory, self.budget)
            # Clean up the array used to save all the necessary information to solve the knapsack problem with the GA algo
            self.greedy_memory = []
        elif self.exp_type == 'episode_lambda':
            self.ctl_lambda = 0.01
        else:
            pass
        # Next episode -> next step
        self._reset_step()
        # Set the budget for the episode
        self.budget = self.episode_budgets.pop(0)
        self.rem_budget = self.budget
        self.rem_budget_ratio = 1
        self.budget_spent_t = 0
        self.budget_spent_e = 0
        if self.exp_type not in ('free_lambda', 'free_lambda_eval', 'improved_drlb', 'improved_drlb_eval'):
            self.ROL = self.T # 3. The number of Lambda regulation opportunities left
            self.ROL_ratio = 1
        self.cur_day = 0
        self.cur_min = 0
        self.total_wins += self.rewards_e
        self.total_rewards += self.wins_e
        # Impressions won in each episode
        self.wins_e = 0
        # Clicks won in each episode
        self.rewards_e = 0
        # Dict and Value necessary for learning the RewardNet
        self.reward_net.V = 0
        self.reward_net.S = []

    def _update_step(self):
        """
        Function that is called before transitioning into step t+1 (updates state t)
        """
        self.global_T += 1
        self.t_step += 1
        self.prev_budget = self.rem_budget
        self.rem_budget = self.prev_budget - self.budget_spent_t
        self.budget_spent_e += self.budget_spent_t
        self.rewards_prev_t = self.reward_t
        self.ROL -= 1
        self.BCR = 0 if self.prev_budget == 0 else -((self.rem_budget - self.prev_budget) / self.prev_budget)
        if self.exp_type in ('improved_drlb', 'improved_drlb_eval'):
            self.CPI = 0 if self.wins_t == 0 else (self.cost_t / self.wins_t) / 300
            self.rewards_prev_t_ratio = 1 if self.possible_clicks_t == 0 else self.reward_t / self.possible_clicks_t
            self.ROL_ratio = self.ROL / self.T
            self.rem_budget_ratio = self.rem_budget / self.budget
        else:
            self.CPM = 0 if self.wins_t == 0 else ((self.cost_t / self.wins_t) * 1000)
        self.WR = self.wins_t / self.imp_opps_t
        # Adaptive eps-greedy policy
        self.eps = max(0.95 - self.anneal * self.global_T, 0.05)

    def _reset_step(self):
        """
        Function to call every time a new time step is entered.
        """
        self.possible_clicks_t = 0
        self.total_rewards_t = 0
        self.reward_t = 0
        self.cost_t = 0
        self.wins_t = 0
        self.imp_opps_t = 0
        self.BCR = 0
        if self.exp_type in ('improved_drlb', 'improved_drlb_eval'):
            self.CPI = 0
        else:
            self.CPM = 0
        self.WR = 0
        self.budget_spent_t = 0
    
    def _update_reward_cost(self, bid, reward, potential_reward, cost, win):
        """
        Internal function to update reward and action to compute the cumulative
        reward and cost within the given step.
        """
        self.possible_clicks_t += potential_reward
        if win:
            self.budget_spent_t += cost
            self.wins_t += 1
            self.wins_e += 1
            self.total_wins += 1
            self.reward_t += reward
            self.rewards_e += reward
            self.total_rewards += reward
            self.cost_t += cost

    def _model_upd(self, eval_mode):
        if not eval_mode:
            self.reward_net.step() # update reward net

        next_state = self._get_state() # observe state s_t+1 (state at the beginning of t+1)
        # get action a_t+1 (adjusting lambda_t to lambda_t+1) from the adaptive greedy policy
        a_beta = self.dqn_agent.act(next_state, eps=self.eps, eval_mode=eval_mode)
        self.ctl_lambda *= (1 + self.BETA[a_beta])

        if not eval_mode:
            # updates for the RewardNet
            sa = np.append(self.cur_state, self.BETA[self.dqn_action]) #self.dqn_action) # state-action pair for t
            self.rnet_r = float(self.reward_net.act(sa)) # get reward r_t from RewardNet
            self.reward_net.V += self.reward_t
            self.reward_net.S.append((self.cur_state, self.BETA[self.dqn_action]))

            # Store in D1 and sample a mini batch and perform grad-descent step
            self.dqn_agent.step(self.cur_state, self.dqn_action, self.rnet_r, next_state)

        self.cur_state = next_state # set state t+1 as state t
        self.dqn_action = a_beta # analogously with the action t+1

    def act(self, obs, eval_mode):
        """
        This function gets called with every bid request.
        By looking at the weekday and hour to progress between the steps and
        episodes during training.
        Returns the bid decision based on the scaled version of the
        bid price using the DQN agent output.
        """
        # within the time step
        if obs['min'] == self.cur_min and obs['weekday'] == self.cur_day:
            pass
        # within the episode, changing the time step
        elif obs['min'] != self.cur_min and obs['weekday'] == self.cur_day:
            self._update_step()
            self._model_upd(eval_mode)
            self.cur_min = obs['min'] 
            # save history
            self.step_memory.append([self.global_T, int(self.rem_budget), self.ctl_lambda, self.eps, self.dqn_action, self.dqn_agent.loss, self.rnet_r, self.reward_net.loss])
            self._reset_step()
        # transition to next episode
        elif obs['weekday'] != self.cur_day:
            self._update_step()
            self._model_upd(eval_mode)
            self.step_memory.append([self.global_T, int(self.rem_budget), self.ctl_lambda, self.eps, self.dqn_action, self.dqn_agent.loss, self.rnet_r, self.reward_net.loss])
            # Updates for the RewardNet at the end of each episode (only when training)
            if not eval_mode:
                for (s, a) in self.reward_net.S:
                    sa = tuple(np.append(s, a))
                    max_r = max(self.reward_net.get_from_M(sa), self.reward_net.V)
                    self.reward_net.add_to_M(sa, max_r)
                    self.reward_net.add(sa, max_r)
            print("Episode Result with Step={} Budget={} Spend={} impressions={} clicks={}".format(self.global_T, int(self.budget), int(self.budget_spent_e), self.wins_e, self.rewards_e))
            # Save history
            self.episode_memory.append([self.budget, int(self.budget_spent_e), self.wins_e, self.rewards_e])
            self._reset_episode() 
            self.cur_day = obs['weekday']
            self.cur_min = obs['min']

        self.imp_opps_t += 1
        bid = self.calc_bid(obs['pCTR'])

        if self.exp_type == 'vanilla_drlb':
            self.greedy_memory.append([obs['pCTR'], obs['payprice'], obs['pCTR']/max(obs['payprice'], 1)])

        return bid

    def calc_bid(self, imp_value):
        # Calculate the theoretically optimal bid
        bid_amt = round(imp_value/self.ctl_lambda, 2)

        curr_budget_left = self.rem_budget - self.budget_spent_t

        if bid_amt > curr_budget_left:
            bid_amt = curr_budget_left

        return bid_amt

    def calc_greedy(self, items, budget_limit):
        # Borrowed from: https://bitbucket.org/trebsirk/algorithms/src/master/knapsack.py
        # Greedy approximation algorithm (Dantzig, 1957)
        bids = []
        spending = 0
        ctr = 0
        items_sorted = sorted(items, key=itemgetter(2), reverse=True)
        while len(items_sorted) > 0:
            item = items_sorted.pop()
            if item[1] + spending <= budget_limit: # should be item[1], currently adds pCTR instead of price?????
                bids.append(item)
                spending += bids[-1][1]
                ctr += bids[-1][0]
            else:
                break
        ctrs = np.array(bids)[:,0]
        costs = np.array(bids)[:,1]
        # Take the max lambda to be more conservative at the beginning of a time step
        opt_lambda = np.max(np.divide(ctrs, costs))
        return opt_lambda


def main():
    # Instantiate the Environment and Agent
    env = gym.make('AuctionEmulator-v0')
    env.seed(0)
    set_seed()
    agent = RlBidAgent()

    train_budget = env.bid_requests.payprice.sum()/8
    # Set budgets for each episode
    budget_proportions = []
    for episode in env.bid_requests.weekday.unique():
        budget_proportions.append(len(env.bid_requests[env.bid_requests.weekday == episode])/env.total_bids)
    for i in range(len(budget_proportions)):
        budget_proportions[i] = round(train_budget * budget_proportions[i])

    epochs = 400

    for epoch in range(epochs):

        print("Epoch: ", epoch+1)
        obs, done = env.reset()
        agent.episode_budgets = budget_proportions.copy()
        if agent.exp_type in ('free_lambda', 'improved_drlb'):
            agent.ctl_lambda = 0.01
        agent._reset_episode()
        agent.cur_day = obs['weekday']
        agent.cur_hour = obs['hour']
        agent.cur_state = agent._get_state() # observe state s_0

        while not done: # iterate through the whole dataset
            bid = agent.act(obs, eval_mode=False) # Call agent action given each bid request from the env
            next_obs, cur_reward, potential_reward, cur_cost, win, done = env.step(bid) # Get information from the environment based on the agent's action
            agent._update_reward_cost(bid, cur_reward, potential_reward, cur_cost, win) # Agent receives reward and cost from the environment
            obs = next_obs
        print("Episode Result with Step={} Budget={} Spend={} impressions={} clicks={}".format(agent.global_T, int(agent.budget), int(agent.budget_spent_e), agent.wins_e, agent.rewards_e))
        agent.episode_memory.append([agent.budget, int(agent.budget_spent_e), agent.wins_e, agent.rewards_e])

        # Saving models and history
        if ((epoch + 1) % 25) == 0:
            PATH = 'models/model_state_{}.tar'.format(epoch+1)
            torch.save({'local_q_model': agent.dqn_agent.qnetwork_local.state_dict(),
                        'target_q_model':agent.dqn_agent.qnetwork_target.state_dict(),
                        'q_optimizer':agent.dqn_agent.optimizer.state_dict(),
                        'rnet': agent.reward_net.reward_net.state_dict(),
                        'rnet_optimizer': agent.reward_net.optimizer.state_dict()}, PATH)

            f = open('models/rnet_memory_{}.txt'.format(epoch+1), "wb")
            cloudpickle.dump(agent.dqn_agent.memory, f)
            f.close()
            f = open('models/rdqn_memory_{}.txt'.format(epoch+1), "wb")
            cloudpickle.dump(agent.reward_net.memory, f)
            f.close()

            pd.DataFrame(agent.step_memory).to_csv('models/step_history_{}.csv'.format(epoch+1),header=None,index=False)
            agent.step_memory=[]
            pd.DataFrame(agent.episode_memory).to_csv('models/episode_history_{}.csv'.format(epoch+1),header=None,index=False)
            agent.episode_memory=[]

        print("EPOCH ENDED")

    env.close() # Close the environment when done


if __name__ == "__main__":
    main()
