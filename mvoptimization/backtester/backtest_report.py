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
import pdb
import json
from signaltest_report import factor_testing as ft
from backtest import fetch_bt
import quantstats as qs
import analysis as an
import warnings
warnings.filterwarnings('ignore')

def backtest_report(holdings, prices = None, returns = None, bmk_levels = None):

    """
    Generates the html backtest report given holdings and returns dataframe.
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
    
    :param bmk_levels: Benchmark levels if necessary
                       Format required:
                       Date(Index) | Symbol
                       2021-01-01  |  100

    """
    config = configparser.ConfigParser()
    config.read('mvoptimization/data/config.txt')

    qs_title = config['BACKTEST REPORT']['report_title']
    qs_output = config['BACKTEST REPORT']['report_output_path']
    qs_theme_name = config['BACKTEST REPORT']['theme_name']
    qs_benchmark_name = config['BACKTEST REPORT']['report_benchmark_name']

    levels_df, res = fetch_bt(holdings = holdings, returns = returns, prices = prices)
    levels_df.index.name = 'Date'
    levels_df.columns.name = None
    levels_df.columns = ['Levels']
        
    levels = pd.Series(levels_df.values.ravel(), index = levels_df.index, name = levels_df.columns[0])
    returns = levels.pct_change()
    
    print(levels)

    if bmk_levels:
        bmk_levels['Date'] = pd.to_datetime(bmk_levels['Date'])
        bmk_levels.set_index('Date', inplace = True)
        bmk_levels = pd.Series(bmk_levels.values.ravel(), index = bmk_levels.index, name = bmk_levels.columns[0])

    print('Generating report')
    print(qs_title)
    qs.reports.html(returns, benchmark = bmk_levels, corr = None, turnover = None, ic = None, rf_rate = None, title = qs_title, output = qs_output, strategy_name = qs_theme_name, benchmark_name = qs_benchmark_name)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
