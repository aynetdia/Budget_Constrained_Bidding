
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os
class Logistic_Regression():
    df_bid_requests=pd.read_csv("D:/lecture sources/Win2021/information system/BCB_help/data/ipinyou/1458/dummified_train_test.txt", sep="\t")
    df_bid_test=pd.read_csv("D:/lecture sources/Win2021/information system/BCB_help/data/ipinyou/1458/dummified_test_test.txt", sep="\t")
    bf_help=pd.read_csv("D:/lecture sources/Win2021/information system/BCB_help/data/ipinyou/1458/train.log.txt", sep="\t")
    # #print(df_bid_test)
    #training

    df_bid_requests['click']=bf_help['click']
    df_bid_test=pd.DataFrame(df_bid_test.astype(np.uint8))
    df_bid_requests=pd.DataFrame(df_bid_requests.astype(np.uint8))

    X = df_bid_requests.drop(['click'], axis=1)
    Y = df_bid_requests.click

    #df_bid_test = df_bid_test[np.intersect1d(X.columns, df_bid_test.columns)]

    X=X[np.intersect1d(X.columns, df_bid_test.columns)]

    # ########### Train model with Logitstic Regression
    # Logistic Regression: Model fitting
    # print(X.dtypes)
    # logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
    # logreg.fit(X, Y)

    #save models
    # joblib_filename = "logreg_1.sav"
    # joblib.dump(logreg, joblib_filename)

    #
    # ## Check auccuracy score
    # print(logreg.score(X,Y))
    # prob = logreg.predict_proba(X)[:, 1]
    # print(roc_auc_score(Y, prob))

    logreg = joblib.load("logreg.sav")
    #
    # # ## Logistic Regression: Predict CTR
    ctr_pre_train=logreg.predict(X)
    ctr_pre_train_proba=logreg.predict_proba(X)
    df_ctr_train = pd.DataFrame(ctr_pre_train_proba, columns=['ctr_prediction_pro','ctr_prediction'])
    df_ctr_train['click'] =ctr_pre_train
    df_ctr_train.to_csv('D:/lecture sources/Win2021/information system/BCB_help/data/ipinyou/1458/ctr_pred_train.txt', index=False, sep='\t', header=True)

    ctr_pred = logreg.predict(df_bid_test[X.columns])
    ctr_pre_proba = logreg.predict_proba(df_bid_test[X.columns])


    df_ctr_test = pd.DataFrame(ctr_pre_proba, columns=['ctr_prediction_pro_0', 'ctr_prediction_pro_1'])
    df_ctr_test['click']=ctr_pred
    print(df_bid_test)
    df_ctr_test.to_csv('D:/lecture sources/Win2021/information system/BCB_help/data/ipinyou/1458/ctr_pred_test.txt',
                         index=False, sep='\t', header=True)

