"""
    Auction Emulator to generate bid requests from iPinYou DataSet.
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import configparser
import json
import os
import pandas as pd

class AuctionEmulatorEnv(gym.Env):
    """
    AuctionEmulatorEnv can be used with Open AI Gym Env and is used to generate
    the bid requests reading the iPinYou dataset files.
    Toy data set with 100 lines are included in the data directory.
    """
    metadata = {'render.modes': ['human']}

    def _load_config(self):
        """
        Parse the config.cfg file
        """
        cfg = configparser.ConfigParser(allow_no_value=True)
        env_dir = os.path.dirname(__file__)
        cfg.read(env_dir + '/config.cfg')
        self.data_src = cfg['data']['dtype']
        if self.data_src == 'ipinyou':
            self.file_in = env_dir + str(cfg['data']['ipinyou_path'])
        self.metric = str(cfg['data']['metric'])

    def __init__(self):
        """
        Args:
        Populates the bid requests to self.bid_requests list.
        """
        self._load_config()
        self._step = 1
        fields = ['click', 'weekday', 'hour', 'bidid', 'timestamp', 'logtype', 'ipinyouid', 'useragent',
        'IP', 'region', 'city', 'adexchange', 'domain', 'url', 'urlid', 'slotid', 'slotwidth', 'slotheight',
        'slotvisibility', 'slotformat', 'slotprice', 'creative', 'bidprice', 'payprice', 'keypage',
        'advertiser', 'usertag']
        self.bid_requests = pd.read_csv(self.file_in, sep="\t", usecols=fields)
        self.total_bids = len(self.bid_requests)
        self.bid_line = {}

    def _get_observation(self, bid_req):
        observation = {}
        if bid_req is not None:
            for feature in bid_req.index.values:
                observation[feature] = bid_req[feature]
        return observation

    def _bid_state(self, bid_req):
        self.auction_type = 'SECOND_PRICE'
        self.bidprice = bid_req['bidprice']
        self.payprice = bid_req['payprice']
        self.click_prob = bid_req['click']
        self.slotprice = bid_req['slotprice']

    def reset(self):
        """
        Reset the OpenAI Gym Auction Emulator environment.
        """
        self._step = 1
        bid_req = self.bid_requests.iloc[self._step]
        self._bid_state(bid_req)
        first_obs = self._get_observation(bid_req)
        # observation, reward, cost, done
        return first_obs, first_obs['click'], first_obs['payprice'], False

    def step(self, action):
        """
        Args:
            action: bid response (bid_price)
        Reward is computed using the bidprice to payprice difference.
        """
        done = False
        self._step += 1

        next_bid = self.bid_requests.iloc[self._step]
        self._bid_state(next_bid)
        next_obs = self._get_observation(next_bid)
        next_r = next_obs['click']
        next_c = next_obs['payprice']

        if self._step > self.total_bids - 1:
            done = True

        return next_obs, next_r, next_c, done

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass
