"""MC2-P1: Market simulator.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""


import pandas as pd
import numpy as np
import datetime as dt
from util import get_data





def author():
    # return georgia tech user name
    return 'harshal'


def get_leverage(date, df, prices):

    # calculate leverage--> basically calculate debt
    leverage = np.sum(abs(prices.ix[date, 1:].multiply(df.ix[date, 1:]))) / \
               (np.sum(prices.ix[date, 1:].multiply(df.ix[date, 1:])) + df.ix[date, 'cash'])

    return leverage


def get_position_values(cashStocks, prices, symbols):

    posVals = cashStocks.copy()
    # calculate total price for each company's stock for each day
    posVals.ix[:, 1:] = cashStocks.ix[:, 1:] * prices.ix[:, 1:]
    return posVals


def get_cash_stocks(order, prices, symbols, start_val,commision,impact):

    """input:
            -order:  dataframe with index as date and columns:[ symbol,order(buy|sell),shares]
            -prices: dataframe with spy and prices of given company's share
            -symbols:  list of company
            -start_val:  starting value of portfolio
            -commission: fix value to be given to broker
            -impact:amount the price moves against the trader compared to the historical data at each transaction
                    - simplicity treat the market impact penalty as a deduction from your cash balance.
    """

    # dataframe with date as index and total cash and no of shares of each company
    df = pd.DataFrame(0, index=prices.index, columns=['cash'] + symbols)
    # initialize cash with starting value
    df.ix[:, 0] = start_val

    for row in order.itertuples():
        date, sym, tradingSignal, n = row  # extract date, symbol,signal, and no. of shares
        traded_share_value = prices.loc[date, sym] * n
        # Transaction cost
        transaction_cost = commision + impact * traded_share_value
        # Update the number of shares and cash based on the type of transaction done
        # Note: The same asset may be traded more than once on a particular day
        if tradingSignal == "BUY":
            df.loc[date:, sym] = df.loc[date:, sym] +n
            df.loc[date:, "cash"] = df.loc[date:, "cash"] + traded_share_value * (-1.0) - transaction_cost
        else:
            df.loc[date:, sym] = df.loc[date:, sym] - n
            df.loc[date:, "cash"] = df.loc[date:, "cash"] + traded_share_value - transaction_cost
    return df


def compute_portvals(orders_file = "./orders/orders-02.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # order is a dataframe with date as index and parse_dates converts type of date from string to datetime
    order = pd.read_csv(orders_file, index_col='Date', parse_dates=True,
                        na_values=['nan'])

    """Create dataframe prices with symbol/dates/prices relevant to the order"""
    start_date = order.index[0]
    end_date = order.index[-1]
    dates = pd.date_range(start_date, end_date)
    symbols = list(order.ix[:, 0].unique())  # ndarray to list of symbols in order
    prices = get_data(symbols, dates)

    """Create dataframe of cash and deposits in stocks, indexed by date"""

    # get total cash and no of shares for each company
    cashStocks = get_cash_stocks(order, prices, symbols, start_val,commission,impact)
    #calculate value for shares of each company
    posVals = get_position_values(cashStocks, prices, symbols)
    # sum all values(cash + value of share for each day)
    portVals = float(posVals.sum(axis=1))

    return portVals


def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-12.csv"
    sv = 1000000


    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)

    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()