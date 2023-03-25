# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 18:51:30 2023

@author: schoudh
"""

import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from datetime import datetime
import os

def remove_files(reqd_file):
    for fname in os.listdir('mvoptimization/data'):
        if fname.startswith(reqd_file):
            return None
        elif fname.startswith("prices_Nifty50"):
            os.remove(os.path.join('mvoptimization/data', fname))
        else:
            continue
    return None

def check_generate_price_file(date, reqd_file):
    files = os.listdir('mvoptimization/data')
    for file in files:
        if reqd_file in file:
            print(f"pricefile {reqd_file} exists.")
            return reqd_file
    else:
        print('Collecting prices for latest date')
        _ = remove_files(reqd_file)
        os.system('python mvoptimization/etl/python_etl.py')
        return reqd_file
    
#price data information
today = datetime.today().strftime('%Y_%m_%d')
pricefile = ''.join(['prices_Nifty50_',today,'.csv'])
_ = check_generate_price_file(today, pricefile)

# Read in price data
pricedata = pd.read_csv(os.path.join('mvoptimization/data', pricefile))
pricedata.set_index('Date', inplace = True)
print(pricedata.head(4))

import sys
sys.exit()
# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
ef.portfolio_performance(verbose=True)