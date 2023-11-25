# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 18:51:30 2023

@author: schoudh
"""
import os

#os.chdir(r'C:\Users\schoudh\OneDrive - MORNINGSTAR INC\QR\QR\Materials\Projects\pyportfolioopt\meanvarianceoptimization')
import pandas as pd
from mvoptimization.optimizers.efficient_frontier import EfficientFrontier
from mvoptimization.optimizers import calc_covariance
from mvoptimization.optimizers import expected_returns
from mvoptimization.backtester.backtest_report import backtest_report
from datetime import datetime
import numpy as np
import configparser

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
pricedata['Date'] = pd.to_datetime(pricedata['Date']).dt.strftime('%Y-%m-%d')
pricedata.set_index('Date', inplace = True)
pricedata.dropna(axis= 1, how = 'any')

#create in and out of sample data
config = configparser.ConfigParser()
config.read('mvoptimization/data/config.txt')

#benchmark price data and convert to returns
bmk_flag = int(config['DATA']['benchmark_data'])
if bmk_flag:
    bmk_symbol = config['DATA']['benchmark_symbol'] 
    bmk_pricefile = ''.join([f'prices_bmk_{bmk_symbol}_',today,'.csv'])
    bmk_pricedata = pd.read_csv(os.path.join('mvoptimization/data', bmk_pricefile))
else:
    bmk_pricedata = None

#rebal_freq in months and start_date shows start date of out of sample period
rebal_freq = config['BACKTEST']['rebalance_freq']
start_date = config['BACKTEST']['start_date']
number_rebal_periods =  np.ceil((pd.to_datetime('today') - pd.to_datetime(start_date)).days/365)
rebal_dates = pd.date_range(start = start_date, periods = number_rebal_periods, freq = str(rebal_freq) + 'M').strftime('%Y-%m-%d').tolist()

trading_port = pd.DataFrame()
for date in rebal_dates:
    train_period = int(config['TRAINING']['train_period'])
    train_start_date = pd.to_datetime(date) - pd.offsets.BMonthEnd(train_period)
    train_start_date = train_start_date.strftime('%Y-%m-%d')
    sample_pricedata = pricedata.loc[(pricedata.index <= date) & (pricedata.index >= train_start_date)].copy()

    #calculate historical mean returns and covariance
    returns, mu = expected_returns.mean_historical_return(sample_pricedata)
    S = calc_covariance.sample_covariance(returns)

    # Optimize for maximal Sharpe ratio
    weight_bounds_input = (config['OPTIMIZER']['min_weight'], config['OPTIMIZER']['max_weight'])
    ef = EfficientFrontier(mu, S, weight_bounds = weight_bounds_input)
    weights = ef.max_sharpe()
    
    #add securities to trading portfolio
    portfolio = pd.DataFrame(list(weights.items()), columns = ['ticker', 'weight'])
    portfolio['date'] = date 
    trading_port = trading_port.append(portfolio)

#backtesting the generated portfolios
pricereturns, _ = expected_returns.mean_historical_return(pricedata)
trading_port = trading_port[['date', 'ticker', 'weight']]

#backtest
#strategy_name, bmk_name = config['BACKTEST REPORT']['strategy_name'], config['BACKTEST REPORT']['bmk_name']
#strategy_title = strategy_name, benchmark_title = bmk_name
bt_results = backtest_report(trading_port, returns = pricereturns, bmk_levels = bmk_pricedata) 
