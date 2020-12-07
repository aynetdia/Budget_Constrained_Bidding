
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import configparser
import json
import os
import pandas as pd


# split the
# an additional column in like ‘day’ or ‘hour’ with values 00, 15, 30 and 45
class Split:
        def get_time_interval(data):
                time_inv=int(data[11:13])
                if time_inv>=0 and time_inv<15:
                        return ("00")
                elif time_inv >= 15 and time_inv < 30:
                        return ("15")
                elif time_inv >= 30 and time_inv < 45:
                        return ("30")
                elif time_inv >= 45 and time_inv <=60:
                        return ("45")
                else:
                        return(None)


# #bid_requests["usertag"]
# bid_requests["timestamp"]=bid_requests["timestamp"].apply(str)
# bid_requests["minute"]=bid_requests["timestamp"].apply(get_time_interval)
#
#
# print(bid_requests[["timestamp","timeinterval"]])


