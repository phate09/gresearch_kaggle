import pandas as pd
import os
import numpy as np
import logging
# define function to compute log returns
from sklearn.model_selection import train_test_split


def log_return(series, periods=1):
    return -np.log(series).diff(periods=periods)



PATH_TO_ALL_DATA = './g-research-crypto-forecasting'
TEST_SIZE = 0.2
train_df = pd.read_csv(os.path.join(PATH_TO_ALL_DATA, 'train.csv'))
bitcoin_df = train_df.loc[train_df['Asset_ID'] == 1]
bitcoin_df.head()
bitcoin_df=bitcoin_df.iloc[:1000]

'''add bitcoin target'''
period = -15
lret_btc = log_return(bitcoin_df.Close, periods=period)
lret_btc.head()
bitcoin_df.loc["Target"] = lret_btc

'''create test, train dataset'''
logging.info(f"Split data to Train and test")
train, test = train_test_split(bitcoin_df, test_size=TEST_SIZE, random_state=42)
'''concatenate 15 rows'''
timesteps = 6
for i in range(len(bitcoin_df) - timesteps):
    target = bitcoin_df.iloc[i + timesteps]["Target"]
    slice_data = bitcoin_df.iloc[i:i + timesteps]
    for t in range(1,timesteps):
        for c in bitcoin_df.columns[2:10]:
            bitcoin_df[c+str(t)]=bitcoin_df[c]
        bitcoin_df = bitcoin_df.copy()
bitcoin_df.head()