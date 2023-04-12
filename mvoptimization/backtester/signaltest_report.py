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
import seaborn as sns
import pdb
from alphalens.tears import (create_returns_tear_sheet,
create_information_tear_sheet,
create_turnover_tear_sheet,
create_summary_tear_sheet,
create_full_tear_sheet,
create_event_returns_tear_sheet,
create_event_study_tear_sheet)
from alphalens.utils import get_clean_factor_and_forward_returns
import alphalens as al
import quantstats as qs
import json
from scipy.stats import zscore 
import analysis as an
import warnings
warnings.filterwarnings('ignore')
import pdb


class factor_testing:

    def __init__(self, holdings, returns):

        self.s3 = boto3.client('s3')
        self.config = configparser.ConfigParser()
        self.config.read('config.txt')
        self.s3_protocol = 's3://'
        self.s3_dir = self.config['AWS']['s3_staging_dir']
        self.region_name = self.config['AWS']['region_name']
        self.bucket = self.config['AWS']['bucket']
        self.generate_returns = bool(int(self.config['generate']['returns']))
        self.ports = self.s3_protocol + self.bucket + '/' + self.config['DATA']['portfolioPathKey']
        self.levels_file = self.s3_protocol + self.bucket + '/' + self.config['DATA']['levelsPathKey']
        self.confidence_buckets = json.loads(self.config['DATA']['portfolio_bucket'])
        self.rm_db = self.config['DATA']['rm_database']
        self.holdings = holdings.copy()
        if 'Date' in self.holdings.columns.tolist():
            self.holdings.rename({'Date' : 'date'}, inplace = True)

        self.returns = returns.copy()
        #rm database
        self.rm_date = self.config['DATA']['rm_date']
        self.rm_date2 = f"{self.config['DATA']['rm_date2']}/{self.rm_date}"
        #signal testing
        self.factor_col = self.config['SIGNALTESTING']['factor']
        self.factor_name = self.config['DATA']['theme_name'] + ' Theme'

    def returns_to_price(self,returns, rm_date, rm_date2):
        rets2 = returns.pivot_table(columns = "shareclassid", index = "MODELDATE", values='dailyreturn_converted').fillna(0)
        rets2 = rets2/100
        rets2.fillna(0, inplace = True)
        prices = (1+rets2).cumprod()*100
        
        rm_date = rm_date
        rm_date2 = rm_date2
        timeindex = pd.read_parquet(f's3://quant-prod-riskmodel-data-monthly/{rm_date2}_equity_global/output/morn-123456-GlobalRiskModel_timeindex/_timeindex/')
        
        timeindex['CANONICALDATE'] = pd.to_datetime(timeindex['CANONICALDATE'])
        timeindex['MODELDATE'] = pd.to_datetime(timeindex['MODELDATE'])
        dates_dict = dict(timeindex[['MODELDATE', 'CANONICALDATE']].to_numpy())
        
        prices.index = pd.to_datetime(prices.index)
        prices.index = pd.Series(prices.index).replace(dates_dict)
        
        return prices

    def format_prices_holdings(self,prices_df, holdings_df, factor):
        prices_df.index.name = 'date'

        if holdings_df[factor].isna().any():
            print('Holdings data has nan factor scores. Removing nans from the dataset... Continuing with reshaping.')
            holdings_df = holdings_df.loc[~holdings_df[factor].isna()]
        else:
            print('Holdings data has no nan factor scores... Continuing with reshaping.')
            pass

        for ele in holdings_df.loc[~holdings_df.date.isin(prices_df.index.tolist()), 'date'].tolist():
            prices_df.loc[ele] = np.nan
        
        # pdb.set_trace()
        prices_df = prices_df.sort_index()
        prices_df = prices_df.ffill()
        
        date_gap = (holdings_df.date.iloc[-1] - holdings_df.date.iloc[-2])
        final_date = holdings_df.date.max() + date_gap
        
        if final_date in prices_df.index.tolist():
            pass
        else:
            final_date = final_date - pd.tseries.offsets.BDay(1)
        
        
        price_dates = sorted(holdings_df.date.dt.strftime('%Y-%m-%d').unique().tolist() + [final_date.strftime('%Y-%m-%d')])
        prices_df = prices_df.loc[prices_df.index.isin(price_dates)]
        
        factor_df = holdings_df.groupby(['date', 'shareclassid'])[factor].mean()
        
        return prices_df, factor_df

    def get_data(self):

        pass

    def get_ic(self):
        
        print('Formatting data')
        self.prices = self.returns_to_price(self.returns, self.rm_date, self.rm_date2)
        self.prices.index = pd.to_datetime(self.prices.index)
        self.holdings.date = pd.to_datetime(self.holdings.date)
        self.prices, self.factor = self.format_prices_holdings(self.prices, self.holdings, self.factor_col)

        # pdb.set_trace()
        self.factor_data = get_clean_factor_and_forward_returns(
        self.factor,
        self.prices,
        quantiles=int(self.config['SIGNALTESTING']['quantiles']),
        periods=json.loads(self.config['SIGNALTESTING']['periods']), 
        filter_zscore=None,
        max_loss=0.45)

        print('Calculating ic')
        ic = create_information_tear_sheet(self.factor_data)
        return ic
    
    def _set_factor_name(self, factor_name):
        self.factor_name = factor_name
        
    def get_factor_score(self):
        """
        Generates the z-scored factor score for the given factor.
        returns: DataFrame
        """

        self.holdings[self.factor_name] = self.holdings.groupby(['date'])[self.factor_col].transform(lambda x : zscore(x, ddof = 1))
        factor_df = self.holdings[['company_id', 'date', self.factor_name]]

        return factor_df


    def get_timeindex(self):
        """
        Fetches the timeindex table.
        returns: DataFrame
        """
        min_date = (pd.to_datetime(self.holdings.date.min()) - pd.offsets.BDay(1)).strftime('%Y-%m-%d')
        query = f"""
        select *
        from "{self.rm_db}"."riskmodel__timeindex"
        where canonicaldate >= date '{min_date}'
        """
        return aws.query_athena2(query, self.s3_dir, self.region_name)
    
    def get_exposures(self):
        """
        Fetch the exposures for selected companies.

        """

        self.timeindex = self.get_timeindex()
        min_hdate = self.holdings.date.min()
        min_exposure_date = self.timeindex.loc[self.timeindex['canonicaldate'] <= pd.to_datetime(min_hdate), 'timeindex'].max()
        query = f"""
        SELECT r_securityid, timeindex, size_industry_standard, value_growth_industry_standard, volatility_composite_industry_standard, momentum_industry_standard, liquidity_industry_standard, yield_factor, quality 
        FROM "{self.rm_db}"."riskmodel__exposures"
        where r_securityid in {str(tuple(self.holdings.company_id.unique().tolist()))}
        and timeindex >= {min_exposure_date}
        """
        self.exposures = aws.query_athena2(query, self.s3_dir, self.region_name)
        self.exposures = self.exposures.merge(self.timeindex[['modeldate', 'canonicaldate', 'timeindex']], on = 'timeindex')
        self.exposures.rename(columns = {'canonicaldate' : 'can_date', 'r_securityid' : 'company_id'}, inplace = True)

        return self.exposures
    
    def _align_dates(self):
        """
        Filters the exposures dataframe to the same dates as that in holdings after applying offset.
        returns: None
        """

        self.exposures = self.exposures.loc[self.exposures.can_date.isin(self.exposures.groupby(pd.to_datetime(self.exposures.can_date).sort_values().dt.to_period('M'))['can_date'].max())]
        self.exposures['date'] = self.exposures.can_date.transform(lambda x: x + pd.offsets.MonthEnd() if not x.is_month_end else x)
        self.exposures.date = self.exposures.date.dt.strftime('%Y-%m-%d')

    def _factor_matrix(self):
        
        self._align_dates()
        self.factor_matrix = self.factor_score.merge(self.exposures, on =['date', 'company_id'])
    
    def correlation(self):
        """
        Calculate spearman correaltion cross-sectionally and take average across dates

        """
        
        len_dates = self.factor_matrix.date.nunique()
        # import pdb
        # pdb.set_trace()
        for dt in self.factor_matrix.date.sort_values().unique().tolist():
            self.corr_matrix = self.factor_matrix.loc[self.factor_matrix.date == dt, [self.factor_name, 'size_industry_standard','value_growth_industry_standard','volatility_composite_industry_standard','momentum_industry_standard','liquidity_industry_standard','yield_factor','quality']]
            self.corr_matrix = self.corr_matrix.corr('spearman')
            if dt == self.factor_matrix.date.min():
                corr_old = self.corr_matrix.copy()
            else:
                corr = (corr_old + self.corr_matrix)
                corr_old = corr.copy()
        
        self.correlation = corr/len_dates
        return self.correlation
    
    def get_correlation(self):
        """
        Correlation between all factors
        returns: DataFrame
        """
        self.factor_score = self.get_factor_score()
        print('Got factor scores')
        self.exposures = self.get_exposures()
        print('Fetched exposures')
        self._factor_matrix()
        self.corr_ = self.correlation()

        return self.corr_

    

    def get_turnover(self, portfolios, **kwargs):
        """
        Turnover values for the portfolios
        returns: DataFrame
        """
        turnover, _ = an.turnover(portfolios, **kwargs)
        return turnover 

    def factor_attribution(self, method = 'FamaMacbeth'):
        
        if method == 'FamaMacbeth':
            self.returns_fm = (1 + self.returns.pivot_table(values = 'dailyreturn_converted', index = 'MODELDATE', columns = 'shareclassid', fill_value = 0)/100).cumprod()
            self.returns_fm.sort_index(inplace = True)
            self.returns_fm = self.returns_fm.groupby(pd.to_datetime(self.returns_fm.index).strftime('%Y-%m')).tail(1)
            date_dict = dict(zip(self.returns_fm.index.sort_values(), pd.date_range(self.returns_fm.index[0], self.returns_fm.index[-1], freq= 'M')))
            self.returns_fm = self.returns_fm.reset_index()
            self.returns_fm['date'] = self.returns_fm['MODELDATE'].replace(date_dict)
            self.returns_fm.set_index('date', inplace = True)
            self.returns_fm.drop(columns = ['MODELDATE'], inplace = True)
            self.returns_fm = self.returns_fm.pct_change() 

            self.returns_fm = self.returns_fm.shift(-1)
            self.returns_fm = self.returns_fm.dropna(axis = 0, how = 'any')



        # return aws.query_athena(query)

    

# factor_testing().get_clean_factor_forward_returns()

    # if __name__ == "__main__":
        
    #     portfolios = aws.get_s3_file(ports, 'csv')
    #     confidence_buckets = config['DATA']['portfolio_bucket'].split(',')
    #     confidence_buckets = [int(ele) for ele in confidence_buckets]
    #     portfolios = portfolios.loc[portfolios.confidence_scores.isin(confidence_buckets)]
    #     portfolios['Wt'] = portfolios.groupby('date')['company_id'].transform(lambda x: 1/len(x))
        
    #     holdings = portfolios.copy()
    #     returns = aws.get_s3_file(levels_file, 'csv')
    #     prices = returns_to_price(returns, rm_date, rm_date2)

    #     prices, factor = format_prices_holdings(prices, holdings, factor)
        
    #     factor_data = get_clean_factor_and_forward_returns(
    #     factor,
    #     prices,
    #     quantiles=1,
    #     periods=[1], 
    #     filter_zscore=None)

        
    #     qs_title = config['DATA']['report_title']
    #     qs_output = config['DATA']['report_output_path']
    #     qs_theme_name = config['DATA']['theme_name']
    #     print('Generating report')
    #     print(qs_title)
    #     qs.reports.html(returns, benchmark = tme, title = qs_title, output = qs_output, strategy_name = qs_theme_name, benchmark_name = 'M* Global TME')
        