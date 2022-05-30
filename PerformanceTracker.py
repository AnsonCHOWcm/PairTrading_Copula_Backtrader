from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from backtrader import Analyzer
from backtrader.mathsupport import standarddev
import numpy as np


class PortfolioPerformance(Analyzer):
    params = (('riskfreerate', 0.0),('annualizedperiod' , 365),)

    def MaxDrawDown(self , data):

        max_drawdown = 10000
        highest_portfolio = data[0]

        for day in range(len(data)):

            if (data[day]>highest_portfolio) :
                highest_portfolio = data[day]
            elif ((data[day]/highest_portfolio)<max_drawdown):
                max_drawdown = data[day]/highest_portfolio

        return (max_drawdown)

    def __init__(self):
        self.bar_num = 0
        self.portfolio_value = []
        self.bar_ret = []
        self.period_ret = 0
        self.annualized_ret = 0
        self.annualized_std = 0
        self.MDD = 0

    def next(self):
        self.bar_num += 1
        self.portfolio_value.append(self.strategy.broker.getvalue())

    def stop(self):
        self.bar_ret = np.log(np.array(self.portfolio_value[1:])/np.array(self.portfolio_value[:-1]))
        self.period_ret = self.portfolio_value[-1]/self.portfolio_value[0]
        self.annualized_ret = np.log(self.period_ret) * ((self.params.annualizedperiod-1)/(self.bar_num-1))
        self.annualized_std = standarddev(self.bar_ret) * np.sqrt((self.params.annualizedperiod-1))
        self.MDD = self.MaxDrawDown(self.portfolio_value)

    def get_analysis(self):
        return dict(periodreturn=self.period_ret, annualizedreturn=self.annualized_ret, sharperatio=(self.annualized_ret - self.params.riskfreerate/self.annualized_std), MDD=self.MDD)