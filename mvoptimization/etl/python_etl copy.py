import pandas as pd
import configparser
import os
from datetime import datetime,date,time,timedelta
from nsepy import get_history

def get_companies(holdings):
    all_symbols = holdings['Symbol']
    reqd_symbols = all_symbols.loc[~all_symbols.str.contains('NIFTY 50')]
    return reqd_symbols.unique().tolist()

def set_type_start_date(start):
    if isinstance(start, datetime):
        start = start.date()
    elif isinstance(start, str):
        start = datetime.strptime(start, '%Y-%m-%d').date()
    else:
        raise TypeError('Start date should be datetime or str.')
    return start

def get_prices(companies, start, end=datetime.today().date()):
    start = set_type_start_date(start)
    prices = pd.DataFrame()
    import pdb
    pdb.set_trace()
    for ticker in companies:
        print(f"Pulling price for stock: {ticker}")
        prices[ticker] = get_history(ticker,start,end)['Close']

    return prices

config = configparser.ConfigParser()
config.read('mvoptimization/data/config.txt')

holdings_file = config['DATA']['holdings']
holdings = pd.read_csv(os.path.join('mvoptimization/data', holdings_file))
unique_companies = get_companies(holdings)

start = config['DATA']['start_date']
prices = get_prices(unique_companies, start)

prices.to_csv('mvoptimization/data/prices_Nifty50_'+datetime.today().strftime('%Y_%m_%d')+'.csv')