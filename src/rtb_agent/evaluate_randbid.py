import torch
import numpy as np
from rl_bid_agent import RlBidAgent
from model import set_seed
import gym, gym_auction_emulator
import pandas as pd

env = gym.make('AuctionEmulator-v0')
env.seed(0)
np.random.seed(0)

train_data = pd.read_csv('data/ipinyou/1458/train.log.txt', sep="\t")

# Set the budgets for each episode according to Cai et al. (2017)
CPM_train = 1000 * (train_data.payprice.sum()/len(train_data))
budget = CPM_train * (10**-3) * 96
eval_budgets = [budget/2,budget/4,budget/8,budget/16,budget/32]

for e_budget in eval_budgets:

    # start evaluating
    obs, done = env.reset()

    cur_episode = obs['weekday']

    clicks = 0
    imps = 0
    temp_budget = e_budget

    while not done: # iterate through the whole dataset
        bid = np.random.randint(0,300)
        next_obs, cur_reward, _, cur_cost, win, done = env.step(bid) # Get information from the environment based on the agent's action
        if win and cur_cost <= temp_budget:
            imps += 1
            temp_budget -= cur_cost
            if cur_reward > 0:
                clicks += 1
        obs = next_obs
        if obs['weekday'] != cur_episode:
            print("Episode result with clicks={}, impressions={}, budget left={}".format(clicks, imps, temp_budget))
            cur_episode = obs['weekday']
            clicks = 0
            imps = 0
            temp_budget = e_budget
    print("Episode result with clicks={}, impressions={}, budget left={}".format(clicks, imps, temp_budget))


env.close()