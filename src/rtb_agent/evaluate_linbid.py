import torch
from rl_bid_agent import RlBidAgent
from model import set_seed
import gym, gym_auction_emulator
import pandas as pd

env = gym.make('AuctionEmulator-v0')
env.seed(0)

train_data = pd.read_csv('data/ipinyou/1458/train.log.txt', sep="\t")

# # Calculate lambda_0 for the first episode in the test set
# last_train_episode = train_data[train_data['weekday'] == 6]
# items = []
# for obs in last_train_episode.values:
#     items.append([obs[1], obs[25], obs[1]/max(obs[25], 1)])

# Set the budgets for each episode according to Cai et al. (2017)
CPM_train = 1000 * (train_data.payprice.sum()/len(train_data))
budget = CPM_train * (10**-3) * 96
eval_budgets = [budget/2,budget/4,budget/8,budget/16,budget/32]

# Calculate the base bid price for the LinBid baseline
base_bid = train_data.payprice.mean()

seg_pctr = pd.read_csv('data/ctr_seg_test.tsv')


for e_budget in eval_budgets:

    # start evaluating
    obs, done = env.reset()

    cur_episode = obs['weekday']

    i = 0
    clicks = 0
    imps = 0
    temp_budget = e_budget

    while not done: # iterate through the whole dataset
        bid = base_bid * (obs['pCTR']/seg_pctr['1'][i])
        next_obs, cur_reward, _, cur_cost, win, done = env.step(bid) # Get information from the environment based on the agent's action
        if win and cur_cost <= temp_budget:
            imps += 1
            temp_budget -= cur_cost
            if cur_reward > 0:
                clicks += 1
        obs = next_obs
        i += 1
        if obs['weekday'] != cur_episode:
            print("Episode result with clicks={}, impressions={}, budget left={}".format(clicks, imps, temp_budget))
            cur_episode = obs['weekday']
            clicks = 0
            imps = 0
            temp_budget = e_budget
    print("Episode result with clicks={}, impressions={}, budget left={}".format(clicks, imps, temp_budget))


env.close()