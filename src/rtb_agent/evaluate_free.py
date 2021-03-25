import torch
from rl_bid_agent import RlBidAgent
from model import set_seed
import gym, gym_auction_emulator
import pandas as pd

env = gym.make('AuctionEmulator-v0')
env.seed(0)

train_data = pd.read_csv('data/ipinyou/1458/train.log.txt', sep="\t")

# Set the budgets for each episode according to Cai et al. (2017)
CPM_train = 1000 * (train_data.payprice.sum()/len(train_data))
budget = CPM_train * (10**-3) * 96
eval_budgets = [budget/2,budget/4,budget/8,budget/16,budget/32]

for e_budget in eval_budgets:

    # Initialize the bidding agent
    set_seed()
    agent = RlBidAgent()
    agent.ctl_lambda = 0.01

    # Load the saved models
    checkpoint = torch.load('models_prev/model_state_free.tar')
    agent.dqn_agent.qnetwork_local.load_state_dict(checkpoint['local_q_model'])
    agent.dqn_agent.optimizer.load_state_dict(checkpoint['q_optimizer'])
    agent.reward_net.reward_net.load_state_dict(checkpoint['rnet'])
    agent.reward_net.optimizer.load_state_dict(checkpoint['rnet_optimizer'])

    # init the current budget for each episode
    agent.episode_budgets = [e_budget, e_budget, e_budget]

    # start evaluating
    obs, done = env.reset()
    agent._reset_episode()
    agent.cur_day = obs['weekday']
    agent.cur_hour = obs['hour']
    agent.cur_state = agent._get_state() # observe state s_0

    while not done: # iterate through the whole dataset
        bid = agent.act(obs, eval_mode=True) # Call agent action given each bid request from the env
        next_obs, cur_reward, potential_reward, cur_cost, win, done = env.step(bid) # Get information from the environment based on the agent's action
        agent._update_reward_cost(bid, cur_reward, potential_reward, cur_cost, win) # Agent receives reward and cost from the environment
        obs = next_obs
    print("Episode Result with Step={} Budget={} Spend={} impressions={} clicks={}".format(agent.global_T, agent.budget, int(agent.budget_spent_e), agent.wins_e, agent.rewards_e))

env.close()