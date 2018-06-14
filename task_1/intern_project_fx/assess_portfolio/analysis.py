"""Analyze a portfolio.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import math

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    # returns datetime type of list (DateTimeIndex)
    dates = pd.date_range(sd, ed)

    #prices_all contains dataframe conatining columns date and elements of syms
    prices_all = get_data(syms, dates)  # automatically adds SPY


    #prices conains dataframe of columns containing syms
    prices = prices_all[syms]  # only portfolio symbols

    # only SPY, for comparison later
    prices_SPY = prices_all['SPY']

    # we will divide series by first day value in order to normalize data
    normalized_prices_SPY=prices_SPY.divide(prices_SPY[0])

    port_val = prices_SPY

    # we will divide whole dataframe by  first day value in order to normalize data
    normalized_portfolio=prices/prices.ix[0]

    # we will multiply normalized data according to  perecentage of money allocated to different company
    for i in range(len(allocs)):
        normalized_portfolio[syms[i]]=normalized_portfolio[syms[i]].apply(lambda x:x*allocs[i])

    # we will multiply by original money value in order to calculate value on each day
    position_values=normalized_portfolio.apply(lambda x:x*sv)

    # we will sum value of all company for each day in order to calculate total value at end of each day
    total_porfolio_value=position_values.sum(axis=1)
    print(total_porfolio_value)
    # Get portfolio statistics (note: std_daily_ret = volatility)
    normalized_total_portfolio_value=total_porfolio_value.divide(total_porfolio_value[0])

    # Cumulative return are used to determine if prices have moved from the downside to the upside
    # or from the upside to the downside.
    cr=total_porfolio_value[-1]/total_porfolio_value[0]-1

    #subtract the opening price from the closing price. Then, divide the result by the opening price.
    daily_return=total_porfolio_value/total_porfolio_value.shift(1)-1

    # it is simple average of daily returns
    adr=daily_return.mean()

    # Standard deviation is a statistical measure of the degree to which an individual value in
    # a probability distribution tends to vary from the mean of the distribution
    sddr=daily_return.std()

    # print(daily_return)
    # Sharpe ratio is the average return earned in excess of the risk-free rate per unit of  total 	risk.
    k=math.sqrt(sf)
    sr=k*(daily_return-rfr).mean()/sddr
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        compare_portfolio = pd.concat([normalized_total_portfolio_value, normalized_prices_SPY],
                                      keys=['portfolio', 'spy'], axis=1)

        plot_data(compare_portfolio)

        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        pass

    # Add code here to properly compute end value
    ev = sv

    return cr, adr, sddr, sr, ev

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    test_code()
