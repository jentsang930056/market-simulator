

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import get_data, plot_data
import scipy.optimize as sp


# This is the function that will be tested by the autograder  		  	   		  		 		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		  	   		  		 		  		  		    	 		 		   		 		  
def optimize_portfolio(
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        syms=["GOOG", "AAPL", "GLD", "XOM"],
        gen_plot=False,
):
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		  		 		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		  		 		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		  		 		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		  		 		  		  		    	 		 		   		 		  
    statistics.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		  		 		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		  		 		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		  		 		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		  		 		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		  		 		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		  		 		  		  		    	 		 		   		 		  
    """

    # Read in adjusted closing prices for given symbols, date range  		  	   		  		 		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed) #['2008-06-01', '2008-06-02', '2008-06-03', '2008-06-04',...
    prices_all = get_data(syms, dates)  # automatically adds SPY  2008-06-02  127.28  117.89  165.24  87.96  38.76
    prices = prices_all[syms]  # only portfolio symbols  2008-06-02  117.89  165.24  87.96  38.76
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later 2008-06-02    127.28

    # find the allocations for the optimal portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
    # note that the values here ARE NOT meant to be correct for a test case

    def max_sharpe(allocs): #get the max sharpe ratio
        # Calculate portfolio statistics
        risk_free_rate = 0.0
        total_days = 252
        normed = prices / prices.iloc[0]
        alloced = normed*allocs
        #pos_val = alloced*start_val #since already normalized, start value does not matter
        port_val = alloced.sum(axis=1)
        daily_rets = (port_val / port_val.shift(1)) - 1
        daily_rets = daily_rets[1:] #the first is '0'
        cum_ret = (port_val[-1] / port_val[0]) - 1
        adr = (daily_rets - risk_free_rate).mean()
        sddr = daily_rets.std()
        sr = (total_days ** 0.5) * adr / sddr

        # Maximize Sharpe Ratio, so minimize the negative Sharpe Ratio
        return -sr

    num_assets = len(syms)
    init_allocs = np.ones(num_assets) / num_assets # initial guess for allocations (uniform allocation)
    bounds = [(0, 1)] * num_assets # define bounds
    constraints = ({'type': 'eq', 'fun': lambda allocs: 1.0 - np.sum(allocs)}) #define contraints
    result = sp.minimize(max_sharpe, init_allocs, method='SLSQP', bounds = bounds, constraints=constraints) #do the minimize

    #best allocations
    allocs = result.x
    risk_free_rate = 0.0
    total_days = 252
    normed = prices / prices.iloc[0] #normalized prices
    normed_SPY = prices_SPY / prices_SPY.iloc[0] #normalized SPY
    alloced = normed * allocs
    port_val = alloced.sum(axis=1) #portfolio value
    daily_rets = (port_val / port_val.shift(1)) - 1 #daily return
    daily_rets = daily_rets[1:]  # the first is '0'
    cr = (port_val[-1] / port_val[0]) - 1 #cumulative return
    adr = (daily_rets - risk_free_rate).mean() #average daily return
    sddr = daily_rets.std() #Standard deviation of daily return
    sr = (total_days ** 0.5) * adr / sddr #Sharpe ratio


    #generate the plot
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here  		  	   		  		 		  		  		    	 		 		   		 		  
        df_temp = pd.concat([port_val, normed_SPY], keys=["Portfolio", "SPY"], axis=1)
        plt.title("Daily Portfolio Value and SPY")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.plot(df_temp)
        plt.xlim(sd, ed)
        plt.legend(['Portfolio', 'SPY'])
        plt.savefig("images/Figure 1.png")
        plt.close()
        pass

    return allocs, cr, adr, sddr, sr



def test_code():
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		  		 		  		  		    	 		 		   		 		  
    """

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]

    # Assess the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )

    # Print statistics  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")



if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader  		  	   		  		 		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		  	   		  		 		  		  		    	 		 		   		 		  
    test_code()
