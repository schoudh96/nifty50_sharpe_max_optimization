import pandas as pd
from datetime import date
import bt
import warnings
import time
import numpy as np


def convert_log_returns(returns, log_returns):
    """
    Converts returns data to log returns.
    """
    if log_returns:
        return np.log(1 + returns)
    else:
        return returns
    

def convert_prices_to_returns(prices, log_returns):
    """
    Converts prices to returns data. 
    If log_returns flag is set to True, prices are converted to log returns data.

    returns: Daily Return data.
    """

    returns = prices.pct_change().dropna(how = 'all')
    if log_returns:
        returns = convert_log_returns(returns, log_returns)
    
    return returns

def fetch_bt(holdings, prices = None, returns = None, rebal_freq = 1):
    """
    Computes the backtest for given portfolios.
    Either of prices or returns along with holdings should be provided.
    
    :param holdings: holdings DataFrame has datewise portfolios
                        DataFrame format required
                        | date | secid | weight |
    
    :param prices: Prices for the given securities
                   Format required: 
                   Date(Index) | Ticker 1 | Ticker 2 | ... |
                   2021-01-01  |  2.25    |  3.5     | ... |

    :param returns: Returns for the given securities 
                    Same format as prices
    """

    

    holdings.columns = ['date', 'secid', 'weight']
    holdings = holdings.pivot_table(index = 'date', columns = 'secid', values = 'weight', aggfunc = 'sum')
    
    if not prices.empty:
        returns = convert_prices_to_returns(prices, False)
        
    returns = returns.fillna(0)

    if len(returns.columns) > len(holdings.columns):
        returns = returns[holdings.columns.tolist()]
    elif len(returns.columns) < len(holdings.columns):
        warn_string = "Price data not available for all holdings! Missing returns for {} securities".format(len(holdings.columns) - len(returns.columns))
        holdings = holdings[returns.columns.tolist()]
        warnings.warn(warn_string)

    levels = (1+returns).cumprod()*100

    for ele in holdings.index[~holdings.index.isin(levels.index.tolist())].tolist():
        levels.loc[ele] = np.nan
    
    levels = levels.sort_index()
    levels = levels.ffill()

    all_dates = levels.index.tolist()
    rebal_dates = holdings.index.tolist()[::rebal_freq]
    
    all_dates = all_dates[all_dates.index(min(rebal_dates)):]
    aum = [100]

    shares_df = pd.DataFrame()
    for dt in all_dates:

        
        if dt == min(all_dates):
            
            shares = aum[0]*holdings.loc[dt]/levels.loc[dt]
            shares_df = shares_df.append(shares)
        elif dt in rebal_dates:
            
            aum.append((shares*levels.loc[dt]).sum())
            print(f'aum is {aum[-1]}')
            shares = aum[-1]*holdings.loc[dt]/levels.loc[dt]
            shares_df = shares_df.append(shares)
            
        else:
            aum.append((shares*levels.loc[dt]).sum())

    

    aum_df = pd.DataFrame(zip(all_dates,aum))
    aum_df[0] = pd.to_datetime(aum_df[0])
    aum_df.set_index(0, inplace = True)
    strat = bt.Strategy('aum', [bt.algos.RunDaily(),
                       bt.algos.SelectAll(),
                       bt.algos.WeighEqually(),
                       bt.algos.Rebalance()])

    port = bt.Backtest(strat, aum_df)

    res = bt.run(port)

    return aum_df, res