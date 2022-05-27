from __future__ import (absolute_import, division, print_function, unicode_literals)
from Copula_Pair_Trading_backtrader_Strategy_class import *
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


if __name__ == '__main__' :

    # Flag to indicating whether it is optimization process

    optimize = False

    print_log = False

    # Run the following code if it is optimization of parameters

    if optimize :

        cerebro = bt.Cerebro()

        sample_window_range = [30 *24 , 60*24 , 90*24 ,120*24 ,150*24 ,180*24]

        cerebro.optstrategy(CopulaStrat , sample_window = sample_window_range , printlog = False , optimize = True)

        cerebro.broker.setcash(3000.0)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.broker.set_slippage_perc(0.001)
        cerebro.broker.addcommissioninfo(CommInfoFractional())

        path = '/Users/ccm/Desktop/Trading/Algo/Data/Crypto/Hourly/*'

        for fname in glob.glob(path):
            datafile = pd.read_csv(fname, index_col=0, parse_dates=True, infer_datetime_format=True)

            data = bt.feeds.PandasData(
                dataname=datafile,
                timeframe=bt.TimeFrame.Minutes,
                compression=60,
                fromdate=datetime.datetime(2020, 11, 1),
                todate=datetime.datetime(2022, 4, 30),
            )

            filename_split = fname.split('/')
            name = filename_split[-1]
            name = name.split('.')

            cerebro.adddata(data, name=name[0])

            cerebro.resampledata(data, timeframe = bt.TimeFrame.Minutes, compression = 60)

        for data in cerebro.datas:

            data.plotinfo.plot = False

        cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name = 'tradeanalyzer')
        cerebro.addanalyzer(bt.analyzers.Returns, _name = 'ret' ,tann = 365*24)

        result = cerebro.run(maxcpus=1)

        strats = [x[0] for x in result]

        for i, strat in enumerate(strats):

            print('Sample Window: %i' % sample_window_range[i])

            print('**********Strategy Performance**********')

            ret = strat.analyzers.ret.get_analysis()

            print('Total Period (Log) Return : %f' % ret['rtot'])

            print('Annualized (Log) Return : %f' % ret['rnorm'])

            sharpe = strat.analyzers.sharpe.get_analysis()

            print('Sharpe Ratio: %.2f' % sharpe['sharperatio'])

            MDD = strat.analyzers.drawdown.get_analysis().max.drawdown

            print('MDD: %.2f' % MDD)

            print('Calmar Ratio : %.2f' % (ret['rnorm'] / (MDD / 100)))

            print('**********End**********')


    # Run the Following code when it is a single backtest

    else :

        cerebro = bt.Cerebro()

        cerebro.addstrategy(CopulaStrat, sample_window = 360*24 ,printlog = print_log)

        cerebro.broker.setcash(3000.0)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.broker.set_slippage_perc(0.001)
        cerebro.broker.addcommissioninfo(CommInfoFractional())

        path = '/Users/ccm/Desktop/Trading/Algo/Data/Crypto/Hourly/*'

        for fname in glob.glob(path):

            datafile = pd.read_csv(fname, index_col=0, parse_dates=True, infer_datetime_format=True)

            data = bt.feeds.PandasData(
                dataname=datafile,
                timeframe=bt.TimeFrame.Minutes,
                compression=60,
                fromdate=datetime.datetime(2020, 11, 1),
                todate=datetime.datetime(2022, 4, 30),
            )

            filename_split = fname.split('/')
            name = filename_split[-1]
            name = name.split('.')


            cerebro.adddata(data , name = name[0])

        for data in cerebro.datas:
            data.plotinfo.plot = False

        cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
        cerebro.addanalyzer(bt.analyzers.Returns, _name = 'ret' ,tann = 365*24)

        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

        strat = cerebro.run(maxcpus=1)

        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

        print('*******Strategy Performance**********')

        ret = strat[0].analyzers.ret.get_analysis()

        print('Total Period (Log) Return : %f' % ret['rtot'])

        print('Annualized (Log) Return : %f' % ret['rnorm'])

        sharpe = strat[0].analyzers.sharpe.get_analysis()

        print('Sharpe Ratio: %.2f' %sharpe['sharperatio'])

        MDD = strat[0].analyzers.drawdown.get_analysis().max.drawdown

        print('MDD: %.2f' %MDD)

        print('Calmar Ratio : %.2f' % (ret['rnorm'] / (MDD/100)))

        tradeanalysis = strat[0].analyzers.tradeanalyzer.get_analysis()

        print('Total Trade : %f' % tradeanalysis.total.total)
        print('Total PnL : %f' % tradeanalysis.pnl.net.total)
        print('Win Rate : %f' % (int(tradeanalysis.won.total) / int(tradeanalysis.total.total)))
        print('Average Win Trade Profit : %f' % tradeanalysis.won.pnl.average)
        print('Average Loss Trade Profit : %f' % tradeanalysis.lost.pnl.average)
        print('Average Trading Period : %f' % tradeanalysis.len.average)

        print('********End**********')

        cerebro.plot()