from __future__ import (absolute_import, division, print_function, unicode_literals)
from Copula_Pair_Trading_backtrader_Strategy_class import *
import glob

import datetime
import backtrader as bt

class TestStrat(bt.Strategy):

    def __init__(self):
        self.price = self.datas[0].close
        self.flag = True
        self.index = 0

    def next(self):

        self.index = self.index + 1

        if self.index == 2:
            print(self.datas[0].datetime.date(0))
            print(self.datas[0].close[np.array(range(1,10))])

        if self.flag:

            print(self.datas[0].datetime.date(0))
            print(self.datas[0].close[-2])

            for i , d in enumerate(self.datas):

                print(i)
                print(d._name)
                print(d.volume[0])

            self.flag = False


if __name__ == '__main__' :

    cerebro = bt.Cerebro()

    path = '/Users/ccm/Desktop/Trading/Algo/Data/Crypto/*'

    for fname in glob.glob(path):

        data = bt.feeds.YahooFinanceCSVData(
            dataname = fname,
            reverse = False
        )

        cerebro.adddata(data)

    cerebro.addstrategy(CopulaStrat)

    cerebro.run()