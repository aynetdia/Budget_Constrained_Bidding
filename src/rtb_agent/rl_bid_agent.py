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
        # self.budget = int(cfg['agent']['budget'])
        self.T = int(cfg['rl_agent']['T']) # T number of timesteps
        self.STATE_SIZE = int(cfg['rl_agent']['STATE_SIZE'])
        self.ACTION_SIZE = int(cfg['rl_agent']['ACTION_SIZE'])
        
    
    def __init__(self):
        self._load_config()
        # Control parameter used to scale bid price
        self.BETA = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]
        self.eps = 0.9
        self.anneal = 2e-5
        self.t_step = 0
        self.global_T = 0
        self.episode_budgets = None
        self.budget = None
        self.step_memory = []
        self.episode_memory = []
        # DQN Network to learn Q function
        self.dqn_agent = DQN(state_size = 7, action_size = 7)
        # Reward Network to learn the reward function
        self.reward_net = RewardNet(state_action_size = 8, reward_size = 1)
        self.dqn_action = 3 # first action - no adjustment

    def _reset_episode(self):
        """
        Function to reset the state when episode changes
        """
        # Lambda sequential regulation parameter
        self.ctl_lambda = 0.0005 if self.budget is None else self.calc_greedy(self.greedy_memory, self.budget)
        print("LAMBDA: ", self.ctl_lambda)
        self.greedy_memory = []
        self._reset_step()
        self.budget = self.episode_budgets.pop(0)
        self.rem_budget = self.budget
        self.budget_spent = 0
        self.ROL = self.T # 3. the number of Lambda regulation opportunities left
        self.cur_day = 0
        self.cur_min = 0
        self.wins_e = 0
        self.rewards_e = 0
        self.reward_net.V = 0
        self.reward_net.S = []
        self.total_wins = 0
        self.total_rewards = 0
        self.rewards_prev_t = 0
        self.total_cost = 0

    def _update_step(self):
        """
        Function that is called before transitioning into step t+1 (updates state t)
        """
        self.global_T += 1
        self.t_step += 1
        self.prev_budget = self.rem_budget # Bt-1
        self.rem_budget = self.budget - self.budget_spent # Bt (2. the remaining budget at time-step t)
        self.ROL -= 1
        self.BCR = 0 if self.prev_budget == 0 else ((self.rem_budget - self.prev_budget) / self.prev_budget)
        self.CPM = 0 if self.wins_t == 0 else ((self.cost_t / self.wins_t) * 1000) # cost of the won ads divided by the number of impressions (expressed in thousands)
        self.WR = self.wins_t / self.imp_opps_t
        self.rewards_prev_t = self.reward_t # 7. total clicks at timestep t-1

    def _reset_step(self):
        """
        Function to call every time a new time step is entered.
        """
        self.reward_t = 0
        self.cost_t = 0
        self.wins_t = 0
        self.imp_opps_t = 0
        self.BCR = 0                      # 4. Budget consumption rate
        self.CPM = 0                      # 5. Cost per mille of impressions between t-1 and t:
        self.WR = 0                       # 6. total wins / total_impression opportunities
        self.eps = max(0.95 - self.anneal * self.t_step, 0.05)
    
    def _update_reward_cost(self, bid, reward, cost, win):
        """
        Internal function to update reward and action to compute the cumulative
        reward and cost within the given step.
        """
        if win == True:
            self.budget_spent += bid
            self.wins_t += 1
            self.wins_e += 1
            self.total_wins += 1
            self.reward_t += reward # click
            self.rewards_e += reward
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

    def act(self, obs, eval_flag=False):
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
            self.reward_net.step(eval_flag) # update reward net
            next_state = self._get_state() # observe state s_1 (state at the beginning of t=1)
            # get action a_1 (adjusting lambda_0 to lambda_1) from the adaptive greedy policy
            a_beta = self.dqn_agent.act(next_state, eps=self.eps) # lambda adjustment for t
            self.ctl_lambda *= (1 + self.BETA[a_beta]) # adjust lambda t-1 to t (0 to 1)
            sa = np.append(self.cur_state, self.dqn_action) # state-action pair for t=0
            rnet_r = float(self.reward_net.act(sa)) # get reward r_0 from RewardNet
            self.reward_net.V += self.reward_t
            self.reward_net.S.append((self.cur_state, self.dqn_action))
            # call agent step
            # Store in D1 and sample a mini batch and perform grad-descent step
            self.dqn_agent.step(self.cur_state, self.dqn_action, rnet_r, next_state, eval_flag)
            self.cur_state = next_state # set state t+1 as state t (in order call it during next transition)
            self.dqn_action = a_beta # analogously with the action t+1
            self.cur_min = obs['min'] 
            # save state, action, lambda, epsilon, rewardnet history here
            self.step_memory.append([self.global_T, self.rem_budget, self.ctl_lambda, self.eps, a_beta, self.dqn_agent.loss, rnet_r, self.reward_net.loss])
            self._reset_step()
        # transition to next episode
        elif obs['weekday'] != self.cur_day:
            self._update_step()
            for (s, a) in self.reward_net.S:
                sa = tuple(np.append(s, a))
                max_r = max(self.reward_net.get_from_M(sa), self.reward_net.V)
                self.reward_net.add_to_M(sa, max_r)
                self.reward_net.add(sa, max_r)
            print("Total Impressions won with Step={} Budget={} Spend={} wins={} rewards={}".format(self.global_T, self.budget, self.budget_spent, self.wins_e, self.rewards_e))
            self.episode_memory.append([self.budget, self.budget_spent, self.wins_e, self.rewards_e])
            self._reset_episode() 
            self.cur_day = obs['weekday']
            self.cur_min = obs['min']

        self.imp_opps_t += 1
        bid = self.calc_bid(obs['pCTR'])

        self.greedy_memory.append([obs['pCTR'], obs['payprice'], obs['pCTR']/max(obs['payprice'], 1)])

        return bid

    def calc_bid(self, imp_value):
        
        bid_amt = round(imp_value/self.ctl_lambda, 2)
        budget_left = self.rem_budget - self.budget_spent

        if bid_amt > budget_left:
            bid_amt = budget_left
        return bid_amt

    def calc_greedy(self, items, budget_limit):
        # Borrowed from: https://bitbucket.org/trebsirk/algorithms/src/master/knapsack.py
        bids = []
        spending = 0
        ctr = 0
        items_sorted = sorted(items, key=itemgetter(2), reverse=True)
        while len(items_sorted) > 0:
            item = items_sorted.pop()
            if item[0] + spending <= budget_limit:
                bids.append(item)
                spending += bids[-1][1]
                ctr += bids[-1][0]
            else:
                break
        ctrs = np.array(bids)[:,0]
        costs = np.array(bids)[:,1]
        opt_lambda = np.max(np.divide(ctrs, costs))
        return opt_lambda


def main():
    # Instantiate the Environment and Agent
    env = gym.make('AuctionEmulator-v0')
    env.seed(0)
    agent = RlBidAgent()

    # obs, done = env.reset()
    # train_budget = env.bid_requests.payprice.sum()/4

    # budget_proportions = []
    # for episode in env.bid_requests.weekday.unique():
    #     budget_proportions.append(len(env.bid_requests[env.bid_requests.weekday == episode])/env.total_bids)
    # for i in range(len(budget_proportions)):
    #     budget_proportions[i] = round(train_budget * budget_proportions[i])
    # agent.episode_budgets = budget_proportions
    # agent._reset_episode()
    # agent.cur_day = obs['weekday']
    # agent.cur_hour = obs['hour']
    # agent.cur_state = agent._get_state() # observe state s_0

    epochs = 50

    for epoch in range(epochs):

        print("Epoch: ", epoch+1)

        obs, done = env.reset()
        train_budget = env.bid_requests.payprice.sum()/4

        budget_proportions = []
        for episode in env.bid_requests.weekday.unique():
            budget_proportions.append(len(env.bid_requests[env.bid_requests.weekday == episode])/env.total_bids)
        for i in range(len(budget_proportions)):
            budget_proportions[i] = round(train_budget * budget_proportions[i])
        agent.episode_budgets = budget_proportions
        agent._reset_episode()
        agent.cur_day = obs['weekday']
        agent.cur_hour = obs['hour']
        agent.cur_state = agent._get_state() # observe state s_0

        while not done:
            bid = agent.act(obs) # obs = state
            next_obs, cur_reward, cur_cost, win, done = env.step(bid)
            agent._update_reward_cost(bid, cur_reward, cur_cost, win)
            obs = next_obs
        print("Total Impressions won {} value = {}".format(agent.total_wins, agent.total_rewards))
        if ((epoch + 1) % 10) == 0:
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

    env.close()


if __name__ == "__main__":
    main()
