import pandas as pd
import numpy as np
from datetime import date
from get_returns import *
from backtest import *
import matplotlib.pyplot as plt
from ffn import *
import os
import aws
import configparser
import boto3
import botocore
import pdb
import json
from signaltest_report import factor_testing as ft
import quantstats as qs
import analysis as an
import warnings
warnings.filterwarnings('ignore')

# s3 = boto3.client('s3')
config = configparser.ConfigParser()
config.read('config.txt')
# s3_protocol = 's3://'

# bucket = config['AWS']['bucket']
# generate_returns = bool(int(config['generate']['returns']))

# def cagr(rets):
#     tr = rets.iloc[-1]/rets[0]
#     tp = (rets.index[-1] - rets.index[0]) / np.timedelta64(1, 'Y')
#     cagr = np.power(tr, 1/tp) - 1
#     return cagr

# def levelsFile(bucket, key, generate_returns):
#     try:
#         file = s3.get_object(Bucket = bucket, Key = key)
#         print('Returns File exists')
#         if generate_returns:
#             print('Generating new returns file')
#             new_key = key.split('.')[0] + '_' + pd.to_datetime('today').strftime('%Y-%m-%d_%H-%M') + '.csv'
#             key = new_key
#             config['DATA']['levelsPathKey'] = new_key
#             with open('config.txt', 'w') as configfile:
#                 config.write(configfile)
#         else:
#             print(f"Using existing returns file: {'/'.join([bucket,key])}")
    
#     except:
#         print(f'New Returns File as specified: {key}')
            
#     return s3_protocol + bucket + '/' + key
        
# def fill_holdings(holdings):
#     portfolios2 = holdings.copy()
#     portfolios2['date'] = pd.to_datetime(portfolios2['date'])
#     portfolios2 = portfolios2.set_index('date')

#     portfolios3 = portfolios2.groupby('company_id', as_index=False, group_keys=False)\
#             .apply(lambda d: d.resample('D').ffill())

#     portfolios3 = portfolios3.reset_index()
#     return portfolios3

# ports = s3_protocol + bucket + '/' + config['DATA']['portfolioPathKey']
# levelsKey = config['DATA']['levelsPathKey']
# levels_file = levelsFile(bucket, levelsKey, generate_returns)
# #rm database
# rm_date = config['DATA']['rm_date']
# rm_date2 = f"{config['DATA']['rm_date2']}/{rm_date}"
#signal testing
signaltest = int(config['SIGNALTESTING']['signaltesting']) == 1
#Fama Macbeth Attribution
# attribution = int(config['SIGNALTESTING']['signaltesting']) == 1

if __name__ == "__main__":
    
    qs_title = config['DATA']['report_title']
    qs_output = config['DATA']['report_output_path']
    qs_theme_name = config['DATA']['theme_name']

    # portfolios = aws.get_s3_file(ports, 'csv')
    # portfolios.drop_duplicates(['date', 'company_id'], inplace = True)
    # confidence_buckets = json.loads(config['DATA']['portfolio_bucket'])
    # # confidence_buckets = [int(ele) for ele in confidence_buckets]
    # portfolios = portfolios.loc[portfolios.confidence_scores.isin(confidence_buckets)]
    # portfolios['Wt'] = portfolios.groupby('date')['company_id'].transform(lambda x: 1/len(x))
    
    # portfolios['date'] = pd.to_datetime(portfolios.date).dt.strftime('%Y-%m-%d')

    # holdings = portfolios.copy()
    # holdings.rename(columns = {'date' : 'Date'}, inplace = True)
    # holdings = holdings.pivot_table(index = 'Date', columns = 'shareclassid', values = 'Wt', aggfunc = 'sum')
    # holdings = holdings.fillna(0)
    # holdings = holdings.reset_index()
    
    # try:
    #     returns = aws.get_s3_file(levels_file, 'csv')
    # except:
    #     returns = fetch_returns(holdings.columns[1:].tolist(),rm_date,rm_date2)
    #     returns.to_csv(levels_file)
        
    levels_df, res = fetch_bt(holdings,returns)
    levels_df.index.name = 'Date'
    levels_df.columns.name = None
    levels_df.columns = ['Levels']

    #signal test
    if signaltest:
        signal = ft(portfolios, returns)
        corr_ = signal.get_correlation()
        turnover, _ = an.turnover(portfolios[['date', 'company_id', 'Wt']], groups = None)
        turnover.rename(columns = {'Turnover' : qs_theme_name + ' Turnover'}, inplace = True)
        turnover.set_index('Date', inplace = True)
        daily_portfolios = fill_holdings(portfolios)
        signal2 = ft(daily_portfolios, returns)
        ic = signal2.get_ic()
        ic = ic.dropna(how = 'any')
        ic_mean = ic.mean()
        ic_mean.name = 'IC'
        
    levels = pd.Series(levels_df.values.ravel(), index = levels_df.index, name = levels_df.columns[0])
    returns = levels.pct_change()

    ff_factors = pd.read_csv('F-F_Research_Data_Factors_daily.CSV', header = 3, names = ['date','Mkt-RF', 'SMB' ,'HML' ,'RF']).dropna()
    ff_factors['date'] = pd.to_datetime(ff_factors['date'])
    ff_factors = ff_factors.loc[(ff_factors.date <= levels_df.index.max()) & (ff_factors.date >= levels_df.index.min())]
    rf_ser = (1 + ff_factors.set_index('date')['RF']/100).cumprod()
    rf = cagr(rf_ser)
    
    print(rf)

    tme_levels = pd.read_csv('tme_levels_pr_usd.csv')
    tme_levels['Date'] = pd.to_datetime(tme_levels['Date'])
    tme_levels.set_index('Date', inplace = True)
    tme = pd.Series(tme_levels.values.ravel(), index = tme_levels.index, name = tme_levels.columns[0])

    print('Generating report')
    print(qs_title)
    levels.to_csv(f'{qs_theme_name}_backtest_levels.csv')
    print(turnover)
    qs.reports.html(returns, benchmark = tme, corr = corr_, turnover = turnover, ic = ic_mean, rf_rate = rf, title = qs_title, output = qs_output, strategy_name = qs_theme_name, benchmark_name = 'M* Global TME')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
