from __future__ import (absolute_import, division, print_function, unicode_literals)
from Copula_Pair_Trading_backtrader_Strategy_class import *
from PerformanceTracker import *
import glob
import pandas as pd
import datetime
import backtrader as bt

class CommInfoFractional(bt.CommissionInfo):

    def getsize(self, price, cash):

        return self.p.leverage * (cash / price)

class TestStrat(bt.Strategy):

    def __init__(self):
        self.price = self.datas[0].close
        self.flag = True
        self.index = 0

    def next(self):

        self.index = self.index + 1

        if self.index == 2:
            print(self.position)
            print(not self.position)

if __name__ == '__main__':

    # Flag to indicating whether it is optimization process

    optimize = False

    print_log = True

    if optimize :

        params_set = [30 * 24 , 60 * 24 , 90 * 24 , 120 * 24 , 150 * 24 , 180 * 24 , 210 * 24, 240 * 24, 270 * 24, 300 * 24, 330 * 24, 360 * 24]

    else :

        params_set = [360*24]

    for window in params_set:

        cerebro = bt.Cerebro()

        cerebro.addstrategy(CopulaStrat, sample_window = window, copula_threshold = 0.01, printlog = print_log)

        cerebro.broker.setcash(3000.0)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.broker.set_slippage_perc(0.001)
        cerebro.broker.addcommissioninfo(CommInfoFractional())

        path = '/Users/ccm/Desktop/Trading/Algo/Data/Crypto/CCXT Data/Hourly/*'

        for fname in glob.glob(path):

            datafile = pd.read_csv(fname, index_col=0, parse_dates=True, infer_datetime_format=True)

            data = bt.feeds.PandasData(
                dataname=datafile,
                timeframe=bt.TimeFrame.Minutes,
                compression=60,
                fromdate=datetime.datetime(2018, 6, 1),
                todate=datetime.datetime(2022, 4, 30)
            )

            name = datafile['symbol'][0]

            cerebro.adddata(data, name=name)

        for data in cerebro.datas:
            data.plotinfo.plot = False

        cerebro.addanalyzer(PortfolioPerformance, _name='portfolioperformance', annualizedperiod=365*24)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')

        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

        strat = cerebro.run(maxcpus=1)

        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

        if optimize:
            print('sample_window : %2f' % window)

        print('*******Strategy Performance**********')

        performance = strat[0].analyzers.portfolioperformance.get_analysis()

        print('Total Period (Log) Return : %f' % performance['periodreturn'])

        print('Annualized (Log) Return : %f' % performance['annualizedreturn'])

        print('Sharpe Ratio: %f' % performance['sharperatio'])

        print('MDD: %f' % performance['MDD'])

        print('Calmar Ratio : %f' % (performance['annualizedreturn'] / performance['MDD']))

        tradeanalysis = strat[0].analyzers.tradeanalyzer.get_analysis()

        print('Total Trade : %f' % tradeanalysis.total.total)
        print('Total PnL : %f' % tradeanalysis.pnl.net.total)
        print('Win Rate : %f' % (int(tradeanalysis.won.total) / int(tradeanalysis.total.total)))
        print('Average Win Trade Profit : %f' % tradeanalysis.won.pnl.average)
        print('Average Loss Trade Profit : %f' % tradeanalysis.lost.pnl.average)
        print('Average Trading Period : %f' % tradeanalysis.len.average)

        print('********End**********')

        if not optimize:
            cerebro.plot()