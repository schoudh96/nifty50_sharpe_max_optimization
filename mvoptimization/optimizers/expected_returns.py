import warnings
import pandas as pd
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

    
def mean_historical_return(prices:pd.DataFrame, is_returns:bool = False,
                           log_returns:bool = False, compounding:bool = True,
                           frequency:int = 252) -> pd.Series:
    """
    Calculates the annualised historical returns.

    :param prices : adjusted closing prices of the securities, with row being date and column being ticker or security id
    :param is_returns: boolean flag describing if the data provided is returns, defaults to False.
    :param log_returns: whether to convert returns data to log_returns. If data is prices, function internally will convert prices to log_returns. Flag defaults to False.
    :param compounding: whether to compound returns and return cagr or arithmetic mean
    :param frequency: (optional)frequency of return calculation, defaults to 252

    :return: annualised mean(daily) return for each asset
    :rtype: pd.Series
    """

    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f'Expected DataFrame, received prices of type {type(prices)}.')
    
    if is_returns:
        prices.dropna(how = 'all', inplace = True)
        returns = convert_log_returns(prices, log_returns)
    else:
        returns = convert_prices_to_returns(prices, log_returns)

    if compounding:
        return (returns, (1+returns).prod()**(frequency/len(returns)) - 1)
    else:
        return (returns, returns.mean(axis = 0) * frequency)
    

    