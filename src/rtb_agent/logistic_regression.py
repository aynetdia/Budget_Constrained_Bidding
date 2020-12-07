
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import configparser
import json
import pandas as pd
import numpy as np
import sys
import os
#module_path = os.path.abspath(os.path.join('..'))
#sys.path.append(module_path+"\\budget_constrained_bidding\src\gym-auction_emulator\gym_auction_emulator\envs")
#from split_time_interval import Split
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score


class Logistic_Regression():

    ########### config
    cfg = configparser.ConfigParser(allow_no_value=True)
    ## Place your working directory:
    env_dir = "D:\lecture sources\Win2021\information system\BCB_help\src\gym-auction_emulator\gym_auction_emulator\envs"
    cfg.read(env_dir + '/config.cfg')
    data_src = cfg['data']['dtype']
    if data_src == 'ipinyou':
        file_in = env_dir + str(cfg['data']['ipinyou_path'])
    metric = str(cfg['data']['metric'])
    ## Train set
    fields = ['click', 'weekday', 'hour', 'bidid', 'timestamp', 'logtype', 'ipinyouid', 'useragent',
        'IP', 'region', 'city', 'adexchange', 'domain', 'url', 'urlid', 'slotid', 'slotwidth', 'slotheight',
        'slotvisibility', 'slotformat', 'slotprice', 'creative', 'bidprice', 'payprice', 'keypage',
        'advertiser', 'usertag']
    bid_requests = pd.read_csv(file_in, sep="\t", usecols=fields)
    ## Test set

    bid_test = pd.read_csv("D:/lecture sources/Win2021/information system/BCB_help/data/ipinyou/1458/test.log.txt", sep="\t", usecols=fields)
    bid_test = bid_test.drop(['click'], axis=1)



    ########### TRAIN SET ###########

    ########### Prep data set

    ## Time interval
    #print(module_path+"\budget_constrained_bidding/src/gym-auction_emulator/gym_auction_emulator/envs")
    #print(module_path+"\\budget_constrained_bidding\src\gym-auction_emulator\gym_auction_emulator\envs")
    #bid_requests["timestamp"] = bid_requests["timestamp"].apply(str)
    #bid_requests["minute"] = bid_requests["timestamp"].apply(Split.get_time_interval)

    ## Check data set , check NAs

    bid_requests['usertag'] = bid_requests['usertag'].fillna(0) #replace NAs with 0

    ## Transform datatypes
    def compute_categories_dummy (data, clmn):
        col = data[clmn].astype({clmn:'category'})
        dum = pd.get_dummies(col, prefix=clmn)
        #d = pd.concat([data, dum], axis=1)
        return(dum)

    def split_dummy(data,split_):
        b = data.str.split(split_)
        a = pd.get_dummies(b.apply(pd.Series).stack()).sum(level=0)
        return(a)

    dum_useragent = split_dummy(bid_requests["useragent"], '_')
    dum_slotwidth = compute_categories_dummy(bid_requests, "slotwidth")
    dum_slotheight = compute_categories_dummy(bid_requests, "slotheight")
    dum_slotvisibility = compute_categories_dummy(bid_requests, "slotvisibility")
    dum_city = compute_categories_dummy(bid_requests, "city")
    dum_adexchange = compute_categories_dummy(bid_requests, "adexchange")
    dum_slotformat = compute_categories_dummy(bid_requests, "slotformat")
    #dum_minute = compute_categories_dummy(bid_requests, "minute")
    #dum_usertag= split_dummy(bid_requests["usertag"], ',')
    #dum_usertag = dum_usertag.apply(int)

    ## Merge into one dataframe
    pdList = [bid_requests[['click', 'slotprice']],
              dum_useragent, dum_slotwidth, dum_slotheight, dum_city, dum_adexchange,
              dum_slotvisibility, dum_slotformat]  #dum_usertag, #dum_minute
    df_bid_requests = pd.concat(pdList, axis=1)



    ##############################################

    ########### TEST SET ###########

    ## Time interval
    #print(module_path+"/budget_constrained_bidding/src/gym-auction_emulator/gym_auction_emulator/envs")
    #print(module_path+"\\budget_constrained_bidding\src\gym-auction_emulator\gym_auction_emulator\envs")
    #bid_test["timestamp"] = bid_test["timestamp"].apply(str)
    #bid_test["minute"] = bid_test["timestamp"].apply(Split.get_time_interval)

    ## Check data set , check NAs
    bid_test['usertag'] = bid_test['usertag'].fillna(0) #replace NAs with 0

    ## Transform datatypes
    dum_useragent = split_dummy(bid_test["useragent"], '_')
    dum_slotwidth = compute_categories_dummy(bid_test, "slotwidth")
    dum_slotheight = compute_categories_dummy(bid_test, "slotheight")
    dum_slotvisibility = compute_categories_dummy(bid_test, "slotvisibility")
    dum_city = compute_categories_dummy(bid_test, "city")
    dum_adexchange = compute_categories_dummy(bid_test, "adexchange")
    dum_slotformat = compute_categories_dummy(bid_test, "slotformat")
    #dum_minute = compute_categories_dummy(bid_requests, "minute")
    #dum_usertag= split_dummy(bid_requests["usertag"], ',')
    #dum_usertag = dum_usertag.apply(int)

    ## Merge into one df
    pdList = [bid_test[['slotprice']],
              dum_useragent, dum_slotwidth, dum_slotheight, dum_city, dum_adexchange,
              dum_slotvisibility, dum_slotformat]  #dum_usertag, #dum_minute , #dum_hour , #weekday
    df_bid_test = pd.concat(pdList, axis=1)

     #print(df_bid_test)
    #training

    ## Split train.log.txt into train and validation sets
    X = df_bid_requests.drop(['click'], axis=1)

    y = df_bid_requests.click
    df_bid_test = df_bid_test[np.intersect1d(X.columns, df_bid_test.columns)]
    X=X[np.intersect1d(X.columns, df_bid_test.columns)]

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    ########### Train model with Logitstic Regression
    # Logistic Regression: Model fitting
    logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
    logreg.fit(x_train, y_train)

    ## Logistic Regression: Evaluate model
    y_pred = logreg.predict(x_val)

    ## Check auccuracy score
    print(logreg.score(x_val, y_val))
    prob = logreg.predict_proba(x_val)[:, 1]
    print(roc_auc_score(y_val, prob))

    #  print(df_bid_requests.columns)


    # ## Logistic Regression: Predict CTR
    ctr_pred = logreg.predict(df_bid_test)
    #
    # ## Merge into one dataframe
    df_ctr_pred = pd.DataFrame(ctr_pred, columns=['ctr_prediction'])
    print(df_ctr_pred.value_counts())
    df_test = [df_ctr_pred, bid_test]
    df_test_final = pd.concat(df_test, axis=1)
    print(df_test_final)
    #
    # ## Save results
    df_test_final.to_csv('D:/lecture sources/Win2021/information system/BCB_help/data/ipinyou/1458/ctr_pred.txt', index=False, sep='\t', header=True)
