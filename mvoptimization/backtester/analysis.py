
# coding: utf-8

# In[1]:

## helper functions to run coverage analysis and other similar functions 


# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import display, HTML
from scipy import stats
import math
import json
import pickle
import os 
from datetime import timedelta
import glob

import warnings
warnings.filterwarnings("ignore") 

# get_ipython().magic('matplotlib inline')


# In[3]:

def build_dir(path, sub_p):
    
    #build folder
    try:

        dir_name = os.path.join(path, sub_p + "_1")
        os.mkdir(dir_name)

    except FileExistsError as e:

        latest_f = list(glob.glob(os.path.join(path, sub_p)+"*"))[-1]

        print(e, "\n Renaming dir name")

        dir_name = latest_f
        tokens = dir_name.split("_")
        tokens[-1] = str(int(tokens[-1]) + 1)
        new_dir = "_".join(tokens)
        print("New dir name: \n", new_dir)
        dir_name = new_dir
        os.mkdir(dir_name)
        
    return dir_name
        
def save(data, dir_name, fname, stype = None):
    
    fname = os.path.join(dir_name, fname)
    
    if stype == 'json':
    
        with open(fname+'.'+stype, 'w') as conf:
            json.dump(data, conf)
        conf.close()

    elif (stype == 'jpg') | (stype == 'png'):
        
        data.savefig(fname+'.'+stype)
        
    elif (stype == 'csv'):
        
        data.to_csv(fname+'.'+stype)
        
    elif (stype == 'xlsx'):
        
        data.to_excel(fname+'.'+stype)

def cov_analysis(df, group_cols, columns = None, dates = None, aggs = 'count', plot = False, save_files = False, dir_name = None):
    """
    Returns dataframe with coverage for each date as reqd
    
    Args:
        *df(dataframe) : dataframe on which to conduct analysis
        *group_cols(list of strs) : cols to group by
        *columns(list of strs) : columns subset 
        *dates(list) : dates subset
        *aggs(str or dict) : dict of different aggs for different columns or if str, then same agg for all columns - default : count 
        *plot(bool) : plot the results if True
    
    """
    if dates:
        
        #subset on dates
        
        rmask = df['date'].isin(dates)
        df = df.loc[rmask, :]

    else:
        
        pass
    
    
    if columns:

        #subset on columns

        cmask = df.columns.isin(columns)
        df = df.loc[:, cmask]

    else:

        pass
     
    #grouping and results
    
    results = df.groupby(group_cols).agg(aggs)
    
    if plot:
        
        plt.clf()
        
        #display results as table
        results = results.loc[results.index != results.index.min()]
        display(results.head(20))
        
        if isinstance(aggs, str):
            
            # if there is only one type of aggregation we can display the columns as a line graph
            
            fig, axs = plt.subplots(figsize = (15,5))
            pic0 = results.plot(ax = axs, kind = 'line', title = 'Coverage Analysis', ylabel = 'Count of Securities')
            pic1 = pic0.get_figure()
            save(pic1, dir_name, str(pic0.title), stype = 'png')
            
        elif isinstance(aggs, dict):
            # if dict, we can show columns of same aggregation as individual line charts
            df_aggs = pd.DataFrame.from_dict(aggs, orient = 'index', columns = ['type'])
            df_aggs['type'] = df_aggs['type'].apply(lambda x: x.__name__ if type(x) != str else x)
            uniq_types = df_aggs['type'].unique().tolist()
            groups = []
            iter2 = 0
            
            for iter1 in uniq_types:
                groups.append(df_aggs[df_aggs.type == iter1].index.tolist())
            
            for group in groups:
                tmp = results.loc[:,group]
                pic0 = tmp.plot(kind = 'line', title = 'Coverage Analysis', ylabel = 'Coverage - Securities', figsize = (15,5))
                pic1 = pic0.get_figure()
                save(pic1, dir_name, str(pic0.title) + "_" + str(groups[iter2]), stype = 'png')
                iter2 += 1
        else:
            
            pass
        
    else:
        pass
    
    return results
    
def densitynplots(df, factor = None, dates = None, agg = 'mean', period_y = 'month', period_x = 'year', save_files = False, dir_name = None):
    """
    Returns density plot across time and box plots
    
    Args:
        *df(dataframe) : dataframe on which to conduct analysis
        *factor(str) : column on which to conduct analysis
        *dates(list) : dates subset
        *agg(str or dict) : func to use for aggregating data into groups in the pivot func 
        *save(bool) : saves the figures and tables in memory in given folder
    
    """
    
    if dates:
        
        #subset on dates
        
        rmask = df['date'].isin(dates)
        df = df.loc[rmask, :]

    else:
        
        pass

     
    #Remove first date for hygiene reasons
    df = df.loc[df.date != df.date.min()]
    
    #removena
    series = df.loc[~(df[factor].isna()), factor]
    
    #create kde plot of entire series irrespective of date
    min_val = series.min()
    max_val = series.max()
    
    print("Density plot of series \n")
    pic_density = series.plot(kind = 'kde', figsize = (15,5), xticks = np.arange(math.floor(min_val)-1,math.ceil(max_val) + 1,2), title = 'Density plot of factor values across time', xlabel = 'z-scores')
    print("type pic dens", str(type(pic_density)))
    save(pic_density.get_figure(), dir_name, 'Density plot of factor values across time', stype = 'png')
    plt.show()
    
    print("Descriptive stats of series across time \n")
    stats = series.describe(percentiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    stats = pd.DataFrame(stats).rename(columns = {0:'value'})
    display(stats)
    save(stats, dir_name, 'Density plot of factor values across time', stype = 'csv')
    
    print("Box plot of series grouped by periods \n")
    df_box = df.set_index(['isin', 'date'])[factor]
    df_box = df_box[~(df_box.isna())]
    df_box = df_box.reset_index()
    
    df_box['date'] = pd.to_datetime(df_box['date'])
    if period_y == 'year':
        df_box[period_y] = df_box.date.dt.year.astype(int)
    elif period_y == 'month':
        df_box[period_y] = df_box.date.dt.month.astype(int)
    elif period_y == 'week':
        df_box[period_y] = df_box.date.dt.week.astype(int)
    else:
        raise TypeError('Period has to be one of year, month or week for boxplot visualization.')
        
    for per in df_box[period_y].sort_values().unique().tolist():
        df_m = df_box[df_box[period_y] == per]
        df_p = pd.pivot_table(df_m, index = 'isin', columns = df_m.date.dt.year, values = factor, aggfunc= 'mean')
        plt.figure()
        pic0 = df_p.plot.box(figsize = (20,5), ylabel = 'Month ' + str(per), grid = True,  title = 'Year wise plots')
        pic1 = pic0.get_figure()
        print("check")
        print(pic0.legend)
        print('check2')
        save(pic1, dir_name, str(pic0.title)+"_"+str(pic0.legend), stype = 'png')
        plt.show()
        
    
    return None

        
def turnover(ports, groups= None, save_plot = False, dir_name = None, plot = False):
    """
    Returns period wise turnover numbers and plot
    
    Args:
        *ports(dataframe) : dataframe of portfolios with weights 
                            - if groups are active add weights for every group
                            
                            expected columns format
                            .........................
                            | Date | SecId | Weight | Groups(optional) |
        
        *groups(str) : group the portfolios and run turnover calc
        *save_plot(bool) : whether to save the turnover plot
        *dir_name(str) : path where to save the plot if save_plot set to True
    
    """
    
    if not groups:
        ports.columns = ['Date', 'SecId', 'wts']  
    else:
        ports.columns = ['Date', 'SecId', 'wts', groups]  
    
    ports['Date'] = pd.to_datetime(ports['Date']).dt.strftime('%Y-%m-%d')
    
    act_dates = ports['Date'].unique().tolist()
    act_dates = sorted(act_dates)
    prev_dates = act_dates.copy()
    prev_dates.pop()
    act_dates = sorted(act_dates, reverse = True)
    act_dates.pop()
    act_dates = sorted(act_dates)
    
    prev_ports = ports.loc[ports['Date'] != ports['Date'].max()]
    dict_dates = dict(zip(prev_dates, act_dates))
    
    prev_ports['Date'].replace(dict_dates, inplace = True)
    
    if not groups:
        
        merge_df = ports.loc[ports['Date'] != ports['Date'].min()].merge(prev_ports,\
                   on = ['Date', 'SecId'], how = 'outer', suffixes = ('_curr', '_prev'))

        merge_df.fillna(0, inplace = True)
        merge_df['sec_turnover'] = np.where(merge_df['wts_curr'] - merge_df['wts_prev'] > 0,\
                                            merge_df['wts_curr'] - merge_df['wts_prev'], 0)

        #merge_df.to_csv(r'merge_df_turnover2.csv', index = False)

        turnover = merge_df.groupby('Date')['sec_turnover'].sum().reset_index(drop = False).\
                   rename(columns = {'sec_turnover' : 'Turnover'})
        
        # display(turnover)
        
        # import pdb
        # pdb.set_trace()
        turnover_tmp = turnover.copy()
        
        if groups:
            turnover_tmp.columns = 'Quantile ' + turnover_tmp.columns.astype(str) 
        else:
            pass
        
        min_date = str(pd.to_datetime(min(turnover_tmp.index)).year)+'-12-31'
        max_date = str(pd.to_datetime(max(turnover_tmp.index)).year)+'-01-01'

        turnover_tmp.set_index('Date', inplace = True)
        turnover_tmp = turnover_tmp[(turnover_tmp.index > min_date) & (turnover_tmp.index < max_date)]
        turnover_tmp.columns = 'Quantile ' + turnover_tmp.columns.astype(str) 

        turnover_tmp = turnover_tmp.reset_index()
        turnover_tmp['Date'] = pd.to_datetime(turnover_tmp['Date'])
        turnover_tmp['Year'] = turnover_tmp['Date'].dt.year
        turn_y = turnover_tmp.groupby('Year').mean()
        turn_avg = turn_y.mean()*100
        turn_avg.name = 'Avg. Ann. Turnover %'
        
        ann_turn = pd.DataFrame(turn_avg)
        ann_turn.index.name = 'Groups'
        if plot:
            ann_turn.plot(figsize = (15,10), kind = 'bar', rot = 0, xlabel = 'Groups', ylabel = 'Avg. Ann. Turnover %', legend = False)
            
        
            #year_turnover = turnover.groupby(pd.to_datetime(ports['Date']).dt.year)['Turnover'].sum().reset_index(drop = False).\
            #           rename(columns = {'sec_turnover' : 'Annual_Turnover'})
            #display(year_turnover)


            pic_turnover = turnover.set_index('Date').plot(figsize = (20,15), grid = True, title = 'Turnover number plot')

            turn_plot = pic_turnover.get_figure()
    
    else:
        
        merge_df = ports.loc[ports['Date'] != ports['Date'].min()].merge(prev_ports,\
                   on = ['Date', 'SecId', groups], how = 'outer', suffixes = ('_curr', '_prev'))
        
        #merge_df.to_csv(r'merge_df_turnover3.csv', index = False)
        
        merge_df['wts_curr'].fillna(0, inplace = True)
        merge_df['wts_prev'].fillna(0, inplace = True)
        
        merge_df['sec_turnover'] = np.where(merge_df['wts_curr'] - merge_df['wts_prev'] > 0,\
                                            merge_df['wts_curr'] - merge_df['wts_prev'], 0)

        turnover = merge_df.groupby(['Date', groups])['sec_turnover'].sum().reset_index(drop = False).\
                   rename(columns = {'sec_turnover' : 'Turnover'})
        
        turnover2 = pd.pivot_table(turnover, index = 'Date', columns = groups, values = 'Turnover', aggfunc = 'mean')
        
        #display(turnover2)
        
        turnover_tmp = turnover2.copy()
        
        if groups:
            turnover_tmp.columns = 'Quantile ' + turnover_tmp.columns.astype(str) 
        else:
            pass
        
        min_date = str(pd.to_datetime(min(turnover_tmp.index)).year)+'-12-31'
        max_date = str(pd.to_datetime(max(turnover_tmp.index)).year)+'-01-01'

        turnover_tmp = turnover_tmp[(turnover_tmp.index > min_date) & (turnover_tmp.index < max_date)]
        turnover_tmp.columns = 'Quantile ' + turnover_tmp.columns.astype(str) 

        turnover_tmp = turnover_tmp.reset_index()
        turnover_tmp['Date'] = pd.to_datetime(turnover_tmp['Date'])
        turnover_tmp['Year'] = turnover_tmp['Date'].dt.year
        turn_y = turnover_tmp.groupby('Year').mean()
        #turn_ys = turn_y.stack()
        turn_avg = turn_y.mean()*100
        turn_avg.name = 'Avg. Ann. Turnover %'
        
        ann_turn = pd.DataFrame(turn_avg)
        ann_turn.index.name = 'Groups'
        
        turnover['Date'] = pd.to_datetime(turnover['Date'])
        
        #cols = 2
        #rows = np.ceil(len(ports[groups].unique())/cols).astype(int)
        rows = np.ceil(len(ports[groups].unique())).astype(int)

        fig, axs = plt.subplots(np.ceil(len(ports[groups].unique())).astype(int) ,figsize=(15,20))
        fig.suptitle('Turnover Plots for different groups', y = 0.91, fontdict={'fontsize' : 20})
        # Make space for and rotate the x-axis tick labels
        #fig.autofmt_xdate()

        #fig2, ax2 = plt.subplots(figsize=(15,15))
        #fig2.suptitle('Single Turnover Plot for different groups')
        
        
        
        counter = 0
        group_list = ports[groups].sort_values().unique().tolist()


        for iter1 in np.arange(0, rows):
            
            if iter1+1 > len(group_list):
                break
            else:
                group = group_list[iter1]
                #print(group)

                turn_ax = turnover.loc[turnover[groups] == group]
                turn_ax.sort_values('Date')
                axs[iter1].plot(turn_ax['Date'], turn_ax['Turnover'])
                axs[iter1].set_title('Group: ' + str(group))
                
                #ax2.plot(turn_ax['Date'], turn_ax['Turnover'], label = str(group))
                #ax2.legend()
                
        plt.show()
        
        for ax in axs.flat:
            ax.set(xlabel='Date', ylabel='Turnover')
    
        display(ann_turn)
        ax_t = ann_turn.plot(figsize = (15,10), kind = 'bar', rot = 0, xlabel = 'Groups', ylabel = 'Avg. Ann. Turnover %', legend = False)
        ax_t.set_title('Annualised Turnover',pad = 15, fontdict={'fontsize':15})
        
        ax_ty = (turn_y*100).plot(figsize = (15,8), rot = 0, xlabel = 'Years' , ylabel = 'Avg. Ann. Turnover %',ylim=[0,100], legend = True)
        ax_ty.set_title('Annual Turnover Plot', pad = 15, fontdict={'fontsize':15})
        
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    #for ax in axs.flat:
    #    ax.label_outer()
    
    #plt.show()
            
    '''save plot doesn't work for groups yet'''

    if save_plot:
        save(turn_plot, dir_name, str(pic_turnover.title), stype = 'png')
    
    if not groups:
        return turnover, ann_turn
    else:
        return turnover2, ann_turn
    
def backtest_metrics(levels, portfolio_cols, benchmark_col):
    
#    """
#    Used to find the post backtest metrics for checking efficacy of the strategy
   
#    Args:
#        *levels(DataFrame) - the levels of different strategies and backtest as a single dataframe.
#                             index should be 'date' and sorted in ascending order.
#                             Expected Schema:
#                             | S1 | S2 | S3 | B1 |
                            
#               |   index     | S1  | S2  | B1    |
#                2008-01-01     100   100   100
#                2008-01-02     102   101   101.5
#                     .          .     .     .
#                     .          .     .     .
#                     .          .     .     .
#               ----------------------------------      
#        *portfolio_cols(list) - list of cols in levels which are strategies to test against benchmark.
#        *benchmark_col(str) - col in levels DataFrame which is the benchmark
       
#    """

    returns = levels.pct_change()
    returns = returns.dropna()

    metrics = pd.DataFrame()

    for strat in portfolio_cols:
        active_rets = returns[strat] - returns[benchmark_col]
        te_ann = np.std(active_rets, ddof = 1)*np.sqrt(252)
        time_period = (pd.to_datetime(returns[strat].index.tolist()[-1].strftime('%Y-%m-%d')) - pd.to_datetime(returns[strat].index.tolist()[0].strftime('%Y-%m-%d'))).days/365
        #print(time_period)
        cagr_port = np.power(levels[strat].iloc[-1]/levels[strat].iloc[0],1/time_period)-1
        cagr_bmk = np.power(levels[benchmark_col].iloc[-1]/levels[benchmark_col].iloc[0], 1/time_period) - 1
        ir_strat_ann = cagr_port/te_ann
        metrics[strat] = pd.Series([np.around(cagr_port*100, 2), np.around(te_ann*100,2),np.around(ir_strat_ann,2)], \
                                   index = ['Ann. Return (%)', 'Ann. T.E.(%)', 'IR(ann.)'])
    
    #print('CAGR Bmk: ', cagr_bmk)
    display(metrics)
    
    return metrics
   
    