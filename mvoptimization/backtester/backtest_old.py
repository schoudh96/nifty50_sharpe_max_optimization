import pandas as pd
from datetime import date
import bt
import warnings
import time
import numpy as np


def fetch_bt(holdings_df, returns_df, rebal_freq = 1):
    holdings_df.Date = pd.to_datetime(holdings_df.Date).dt.date
    returns_df.MODELDATE = pd.to_datetime(returns_df.MODELDATE).dt.date
    holdings_df.set_index('Date', inplace = True)
    if holdings_df.columns.tolist()[0][:2] == "0C":
        returns_df_pivot = returns_df.pivot_table(columns = "r_securityid", index = "MODELDATE", values='dailyreturn_converted').fillna(0)
    elif holdings_df.columns.tolist()[0][:2] == "0P":
        returns_df_pivot = returns_df.pivot_table(columns = "shareclassid", index = "MODELDATE", values='dailyreturn_converted').fillna(0)
    returns_df_pivot.max(axis=1).plot(),returns_df_pivot.min(axis=1).plot(), returns_df_pivot.mean(axis=1).plot()
    
    if len(returns_df_pivot.columns) > len(holdings_df.columns):
        returns_df_pivot = returns_df_pivot[holdings_df.columns.tolist()]
    elif len(returns_df_pivot.columns) < len(holdings_df.columns):
        warn_string = "Price data not available for all holdings! Missing returns for {} securities".format(len(holdings_df.columns) - len(returns_df_pivot.columns))
        holdings_df = holdings_df[returns_df_pivot.columns.tolist()]
        warnings.warn(warn_string)
        
    holdings_df = holdings_df[returns_df_pivot.columns.tolist()]

    returns_df_pivot = returns_df_pivot/100
    levels = (1+returns_df_pivot).cumprod()*100

    for ele in holdings_df.index[~holdings_df.index.isin(levels.index.tolist())].tolist():
        levels.loc[ele] = np.nan
    
    levels = levels.sort_index()
    levels = levels.ffill()

    all_dates = levels.index.tolist()
    rebal_dates = holdings_df.index.tolist()[::rebal_freq]
    
    all_dates = all_dates[all_dates.index(min(rebal_dates)):]
    aum = [100]
    
    start = time.time()
    
    shares_df = pd.DataFrame()
    for dt in all_dates:
        
        if dt == min(all_dates):
            
            shares = aum[0]*holdings_df.loc[dt]/levels.loc[dt]
            shares_df.append(shares)
        elif dt in rebal_dates:
            
            start_time_rebal = time.time()
            aum.append((shares*levels.loc[dt]).sum())
            
            shares = aum[-1]*holdings_df.loc[dt]/levels.loc[dt]
            end_time_rebal = time.time()
            shares_df.append(shares)
            
        else:
            aum.append((shares*levels.loc[dt]).sum())

        
    end = time.time()

    
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