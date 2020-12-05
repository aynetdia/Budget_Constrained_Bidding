

import configparser
import json
import os
from typing import Any, Union
import gym
import pandas as pd
import numpy as np


class Random_Bidding(gym.Env):

        metadata = {'render.modes': ['human']}

        def _load_config(self):

                cfg = configparser.ConfigParser(allow_no_value=True)
                ##This line should be changed as configure setting
                env_dir = "D:\lecture sources\Win2021\information system\BCB_help\src\gym-auction_emulator\gym_auction_emulator\envs"
                cfg.read(env_dir + '/config.cfg')

                self.data_src = cfg['data']['dtype']
                if self.data_src == 'ipinyou':
                        self.file_in = env_dir + str(cfg['data']['ipinyou_path'])
                self.metric = str(cfg['data']['metric'])


        def __init__(self):
                self._load_config()
                fields = ['click', 'slotprice', 'payprice']
                self.bid_requests = pd.read_csv(self.file_in, sep="\t", usecols=fields)

                self.total_bids = len(self.bid_requests)
                print(self.total_bids)

       # def step(self):
       #      #   mkt_price = max(self.slotprice, self.payprice)


        def random_bidding(self):

                #generate a random number generator (0,300)
                #get a dataframe with columns('click', 'slotprice', 'payprice','r_bid_price','wins')
                c_names=('click', 'slotprice', 'payprice','r_bid_price','wins')
                zero_Data=np.zeros(shape=(self.total_bids,len(c_names)))
                self.df=pd.DataFrame(zero_Data,columns=c_names)
                self.df['click']=self.bid_requests['click']
                self.df['slotprice'] = self.bid_requests['slotprice']
                self.df['payprice'] = self.bid_requests['payprice']
                self.df['r_bid_price']=np.random.randint(0,100,[self.total_bids,1])

                def wins_value(row):
                        if row['r_bid_price'] >= row['slotprice'] and row['r_bid_price'] > row['payprice']:
                                return 1
                        return 0
                self.df['wins']=self.df.apply(wins_value,axis=1)



def main():
        rb=Random_Bidding()
        rb.random_bidding()
        click1=sum(rb.df['click'])
        wins1=rb.total_bids
        wins2=sum(rb.df['wins'])
        click2=sum(rb.df.loc[rb.df['wins']==1]['click'])
        print(click1)
        print(click2)
        print("Total actual random winning Impressions = {} clicks = {} \n;".format(wins1,click1),
              "Total random winning Impressions = {} clicks = {}".format(wins2, click2))


if __name__ == "__main__":
        main()




