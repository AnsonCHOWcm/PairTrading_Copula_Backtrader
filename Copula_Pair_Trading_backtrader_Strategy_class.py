import backtrader as bt
import numpy as np
import scipy.stats as stats
import scipy
import sys
from statsmodels.distributions.empirical_distribution import ECDF

class CopulaStrat(bt.Strategy):

    params = (
        ('copula_threshold', 0.05),
        ('sample_window', 90),
        ('printlog' , True),
        ('optimize', False),
    )

    def log(self ,txt ,doprint = None):

        dt = self.datas[0].datetime.date(0)
        tm = self.datas[0].datetime.time(0)

        if doprint == None :

            doprint = self.params.printlog

        if doprint :

            print('%s - %s, %s' % (dt.isoformat(), tm.isoformat(), txt))

    def Copula_AIC(self , copula , theta):

        Likelihood = 0

        u = self.first_asset_ECDF(self.first_asset_sample)

        v = self.second_asset_ECDF(self.second_asset_sample)

        n = len(u)

        for i in range(n):

            if u[i] == 0:

                u[i] = 1/n * 1/2

            elif u[i] == 1:

                u[i] = 1 - 1/n * 1/2

            if v[i] == 0:

                v[i] = 1/n * 1/2

            elif v[i] == 1:

                v[i] = 1 - 1/n * 1/2



        # Computing the likelihood of the observed data

        if copula == 'Clayton':

            Likelihood = sum((theta + 1)*(u ** (-1*theta) + v ** (-1*theta) - 1) ** (-2-1/theta) * u ** (-1 * theta - 1) * v ** (-1 * theta - 1))

        elif copula == 'Gumbel':

            A = (-1*np.log(u)) ** theta + (-1*np.log(v)) ** theta

            C = np.exp(-1 * (A) ** 1/theta)

            Likelihood = sum(C * (u * v) ** (-1) * A ** (-2 + 2/theta) * (np.log(u) * np.log(v)) ** (theta - 1) * (1 + (theta - 1) * A ** (-1/theta)))

        elif copula == 'Frank':

            Likelihood = sum((theta * (np.exp(-1 * theta) - 1) * (np.exp(-1 * theta * (u + v))))/(np.exp(-1*theta*u - 1) * np.exp(-1*theta*v - 1) + (np.exp(-1 * theta) - 1)) ** 2)

        return (-2 * Likelihood + 2)

    def CopulaFitting(self , tau):

        theta = np.zeros(3)
        lowest_AIC = pow(2,31) - 1
        selected_copula = ''
        selected_theta = 0

        input_tau = tau

        if tau == 1:

            input_tau = 0.9999

        # Computing the theta for Clayton Copula

        theta[0] = 2 * input_tau * (1-input_tau) ** (-1)

        # Computing the theta for Gumbel Copula

        theta[1] = (1 - input_tau) ** (-1)

        # Computing the theta for Frank Copula

        intrgral = lambda t : t/(np.exp(t)-1)

        frank_fun = lambda theta : (input_tau-1)/4 - 1/theta*(1/theta * scipy.integrate.quad(intrgral , sys.float_info.epsilon , theta)[0] - 1)

        theta[2] = scipy.optimize.minimize(frank_fun , 4 , method = 'BFGS' , tol = 1e-5).x

        # Selecting the best fit Copula by AIC

        for i , copula in enumerate(['Clayton' , 'Gumbel' , 'Frank']) :

            AIC = self.Copula_AIC(copula , theta[i])

            # Finding the Copula with lowest AIC s.t. the Copula fit the data better

            if AIC < lowest_AIC:

                lowest_AIC = AIC

                selected_copula = copula
                selected_theta = theta[i]

        return selected_copula, selected_theta

    def MaginalCopula(self, u, v, copula):

        input_u = u

        input_v = v

        n = self.params.sample_window

        # Prevent the Extreme Case of u,v which is 0 and 1

        if u == 0:

            input_u = 1/n * 1/2

        elif u == 1:

            input_u = 1 - 1/n * 1/2

        if v == 0:

            input_v = 1/n * 1/2

        elif v == 1:

            input_v = 1 - 1/n * 1/2

        # Computing marginal copula of the observed data

        marginal_copula_X_given_Y = 0

        marginal_copula_Y_given_X = 0

        if copula == 'Clayton':

            marginal_copula_X_given_Y = input_v ** (-1 * self.theta -1) * (
                                        input_u ** (-1 * self.theta) + input_v ** (-1 * self.theta) -1) ** (-1/self.theta -1)

            marginal_copula_Y_given_X = input_u ** (-1 * self.theta - 1) * (
                                        input_u ** (-1 * self.theta) + input_v ** (-1 * self.theta) - 1) ** (-1/self.theta - 1)

        elif copula == 'Gumbel':

            A = (-1 * np.log(input_u)) ** self.theta + (-1 * np.log(input_v)) ** self.theta

            C = np.exp(-1 * (A) ** 1/self.theta)

            marginal_copula_X_given_Y = C * (A) ** ((1-self.theta)/self.theta) * (-1*np.log(input_v)) ** (self.theta - 1) * 1/input_v

            marginal_copula_Y_given_X = C * (A) ** ((1-self.theta)/self.theta) * (-1*np.log(input_u)) ** (self.theta - 1) * 1/input_u

        elif copula == 'Frank':

            marginal_copula_X_given_Y = np.exp(-1 * self.theta * input_v) * (np.exp(-1 * self.theta * input_u) -1 )/(np.exp(-1*self.theta)-1 + (np.exp(self.theta*input_u)-1)*(np.exp(self.theta*input_v)-1))

            marginal_copula_Y_given_X = np.exp(-1 * self.theta * input_u) * (np.exp(-1 * self.theta * input_v) -1 )/(np.exp(-1*self.theta)-1 + (np.exp(self.theta*input_u)-1)*(np.exp(self.theta*input_v)-1))

        return marginal_copula_X_given_Y , marginal_copula_Y_given_X

    def __init__(self):

        self.rebalance_month = -1
        # Storing the selected pair and its sample log return
        self.selected_pair = []
        self.first_asset_sample = []
        self.second_asset_sample = []
        #Storing the ECDF of the selected pair
        self.first_asset_ECDF = None
        self.second_asset_ECDF = None
        # Storing the best fitting copula class and its parameter theta
        self.copula =''
        self.theta = 0
        # Storing the hedging ratio for the trading pair as we would like to capture the alpha of divergence from equilibrium
        self.hedge_ratio = 0
        # Storing the portfilio spending
        self.max_pos = 0.98
        # Tracking the day passed as warm-up period
        self.passing_day = 0

    def next(self):

        self.passing_day = self.passing_day + 1

        # Checking the input data at the first day

        if self.passing_day == 1 :

            for data in self.datas :

                print(data._name)

        # Making sure there is enough data for us to sample the tau and ECDF

        if self.passing_day < self.params.sample_window:
            return

        # We would change the selected pair once a month
        if self.rebalance_month != self.datas[0].datetime.date(0).month:
            self.rebalance_month = self.datas[0].datetime.date(0).month

            tau = 0

            # We re-selected the traded pair with highest kendalltau based on last 30 days log_ret

            for i in range(len(self.datas)-1):
                for j in range(i+1 , len(self.datas)):

                    first_asset_price = []
                    second_asset_price = []

                    for day in range(-1 * (self.params.sample_window) , 1):

                        first_asset_price.append(self.datas[i].close[day])
                        second_asset_price.append(self.datas[j].close[day])

                    if sum(np.array(first_asset_price)<0.01) > 0 or sum(np.array(second_asset_price)<0.01) > 0:

                        continue

                    first_asset_log_ret = np.log(first_asset_price[1:]) - np.log(first_asset_price[:-1])

                    second_asset_log_ret = np.log(second_asset_price[1:]) - np.log(second_asset_price[:-1])

                    # Computing the ECDF of the selected sample for estimating the input of Copula

                    first_asset_ECDF = ECDF(first_asset_log_ret)
                    second_asset_ECDF = ECDF(second_asset_log_ret)

                    u = first_asset_ECDF(first_asset_log_ret)
                    v = second_asset_ECDF(second_asset_log_ret)

                    tau_ = stats.kendalltau(u,v)[0]

                    if tau < tau_:

                        tau = tau_

                        self.selected_pair = [self.datas[i]._name , self.datas[j]._name]
                        self.first_asset_sample = first_asset_log_ret
                        self.second_asset_sample = second_asset_log_ret
                        self.first_asset_ECDF = first_asset_ECDF
                        self.second_asset_ECDF = second_asset_ECDF

            self.log('Selected pair %s & %s' %(self.selected_pair[0], self.selected_pair[1]))

            # Computing the hedge ratio for the selected pair

            self.hedge_ratio = (np.cov(self.first_asset_sample , self.second_asset_sample)[0,0]) / np.var(self.second_asset_sample)

            # Selecting the best fit copula class and fitting its parameter theta

            self.copula , self.theta = self.CopulaFitting(tau)

            # Remove Any Position after the Re-selection if the traded pair is not the selected pair

            for data in self.datas:

                if (not (data._name in self.selected_pair) and self.getposition(data).size != 0) :
                    self.close(data)
                    self.log('Close Position : %s' % data._name)

        # Extracting the Current close price of the selected pair

        if len(self.selected_pair) == 0 :
            return

        for data in self.datas:

            if data._name == self.selected_pair[0]:

                u = self.first_asset_ECDF([np.log(data.close[0]) - np.log(data.close[-1])])[0]
                first_asset = data

            elif data._name == self.selected_pair[1]:

                v = self.second_asset_ECDF([np.log(data.close[0]) - np.log(data.close[-1])])[0]
                second_asset = data

        # Computing the Current Marginal Copula to track the equilibrium status of two asset

        marginal_copula_X_given_Y, marginal_copula_Y_given_X = self.MaginalCopula(u, v, self.copula)

        # If there is no position , then we would check the equilibrium situation and generate trading signal

        if self.getposition(first_asset).size == 0:

            first_asset_proportion = 1 / (1+self.hedge_ratio) * self.max_pos

            second_asset_proportion = self.hedge_ratio / (1+self.hedge_ratio) * self.max_pos

            # Do Long and Short based on the divergence of the equilibrium of two correlated asset

            if ((marginal_copula_X_given_Y < self.params.copula_threshold) and (marginal_copula_Y_given_X > 1 - self.params.copula_threshold)):

                self.order_target_percent(data = first_asset, target = first_asset_proportion)
                self.order_target_percent(data = second_asset, target = -1*second_asset_proportion)

                self.log('Long : %s & Short : %s' % (first_asset._name ,second_asset._name))

            elif ((marginal_copula_Y_given_X < self.params.copula_threshold) and (marginal_copula_X_given_Y > 1 - self.params.copula_threshold)):

                self.order_target_percent(data = first_asset, target = -1*first_asset_proportion)
                self.order_target_percent(data = second_asset, target = second_asset_proportion)

                self.log('Long : %s & Short : %s' % (second_asset._name, first_asset._name))

        else:

            # Close the Position if the divergenece disappear

            if ((self.getposition(first_asset).size > 0) and ((marginal_copula_X_given_Y > self.params.copula_threshold) and (marginal_copula_Y_given_X < 1 - self.params.copula_threshold))):
                self.close(first_asset)
                self.close(second_asset)

                self.log('Close Position : Long %s & Short %s' % (first_asset._name ,second_asset._name))

            elif ((self.getposition(first_asset).size < 0) and ((marginal_copula_Y_given_X > self.params.copula_threshold)and (marginal_copula_X_given_Y < 1 - self.params.copula_threshold))):
                self.close(first_asset, exectype=bt.Order.Market)
                self.close(second_asset, exectype=bt.Order.Market)

                self.log('Close Position : Long : %s & Short : %s' % (second_asset._name, first_asset._name))

    def notify_order(self, order):

        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:

            if order.isbuy():
                self.log('BUY EXECUTED, Price : %.2f, Size : %.2f ,Cost : %.2f, Comm %.2f' % (
                         order.executed.price,
                         order.executed.size,
                         order.executed.value,
                         order.executed.comm))

            elif order.issell():
                self.log('SELL EXECUTED, Price : %.2f, Size : %.2f ,Cost : %.2f, Comm %.2f' % (
                         order.executed.price,
                         order.executed.size,
                         order.executed.value,
                         order.executed.comm))

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

    def notify_trade(self, trade):

        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f , NET %.2f' % (trade.pnl, trade.pnlcomm))

    def stop(self):

        if not self.params.optimize:
            return

        self.log('(Sample Window %2d) Ending Value %.2f' %
                 (self.params.sample_window, self.broker.getvalue()), doprint=True)































