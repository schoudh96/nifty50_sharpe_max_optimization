import pandas as pd
import configparser
import os
from datetime import datetime,date,time,timedelta
from nsepy import get_history
import yfinance as yf

def get_companies(holdings):
    all_symbols = holdings['Symbol']
    reqd_symbols = all_symbols.loc[~all_symbols.str.contains('NIFTY 50')]
    return reqd_symbols.unique().tolist()

def set_type_start_date(start):
    if isinstance(start, datetime):
        start = start.replace(hour = 0, minute = 0, second = 0, microsecond=0)
    elif isinstance(start, str):
        start = datetime.strptime(start, '%Y-%m-%d')
    else:
        raise TypeError('Start date should be datetime or str.')
    return start

def check_end_date_type(end):
    if isinstance(end, str):
        end = datetime.strtime(end, '%Y-%m-%d')
    elif isinstance(end, datetime):
        end = end.replace(hour = 0, minute = 0, second = 0, microsecond=0)
    else:
        raise TypeError('End date should be of type datetime or str')

def get_prices(companies, start, end=datetime.today()):
    end = check_end_date_type(end)
    start = set_type_start_date(start)
    prices = pd.DataFrame()
    for ticker in companies:
        print(f"Pulling price for stock: {ticker}")
        ticker = ticker + '.NS'
        company = yf.Ticker(ticker)
        company._tz = 'Asia/Kolkata'
        prices[ticker] = company.history(start=start,end=end)['Close']

    return prices

config = configparser.ConfigParser()
config.read('mvoptimization/data/config.txt')

holdings_file = config['DATA']['holdings']
holdings = pd.read_csv(os.path.join('mvoptimization/data', holdings_file))
unique_companies = get_companies(holdings)

start = config['DATA']['start_date']
prices = get_prices(unique_companies, start)
prices.index = pd.to_datetime(prices.index).strftime('%Y-%m-%d')
prices.to_csv('mvoptimization/data/prices_Nifty50_'+datetime.today().strftime('%Y_%m_%d')+'.csv')

bmk_flag = int(config['DATA']['benchmark_data'])

if bmk_flag: 
    bmk_symbol = config['DATA']['benchmark_symbol'] 
    bmk_prices = get_prices([bmk_symbol], start)
    bmk_prices.index = pd.to_datetime(bmk_prices.index).strftime('%Y-%m-%d')
    bmk_prices.to_csv(f'mvoptimization/data/prices_bmk_{bmk_symbol}_'+datetime.today().strftime('%Y_%m_%d')+'.csv')
