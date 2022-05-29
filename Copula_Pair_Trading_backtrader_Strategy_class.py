import backtrader as bt
import numpy as np
import scipy.stats as stats
import scipy
import sys
from statsmodels.distributions.empirical_distribution import ECDF
import statsmodels as sm

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

            Likelihood = sum(np.log((theta + 1)*(u ** (-1*theta) + v ** (-1*theta) - 1) ** (-2-1/theta) * u ** (-1 * theta - 1) * v ** (-1 * theta - 1)))

        elif copula == 'Gumbel':

            A = (-1*np.log(u)) ** theta + (-1*np.log(v)) ** theta

            C = np.exp(-1 * (A) ** 1/theta)

            Likelihood = sum(np.log(C * (u * v) ** (-1) * A ** (-2 + 2/theta) * (np.log(u) * np.log(v)) ** (theta - 1) * (1 + (theta - 1) * A ** (-1/theta))))

        elif copula == 'Frank':

            Likelihood = sum(np.log((theta * (np.exp(-1 * theta) - 1) * (np.exp(-1 * theta * (u + v))))/(np.exp(-1*theta*u - 1) * np.exp(-1*theta*v - 1) + (np.exp(-1 * theta) - 1)) ** 2))

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

            density_function_input_u = lambda u : (self.theta + 1) * (u ** (-1 * self.theta) + input_v ** (-1 * self.theta) - 1) ** (-2 - 1/self.theta) * u ** (
                        -1 * self.theta - 1) * input_v ** (-1 * self.theta - 1)

            density_function_input_v = lambda v: (self.theta + 1) * (input_u ** (-1 * self.theta) + v ** (-1 * self.theta) - 1) ** (-2 - 1/self.theta) * input_u ** (
                                                         -1 * self.theta - 1) * v ** (-1 * self.theta - 1)

            d_C_d_v = input_v ** (-1 * self.theta -1) * (input_u ** (-1 * self.theta) + input_v ** (-1 * self.theta) -1) ** (-1/self.theta -1)

            d_C_d_u = input_u ** (-1 * self.theta - 1) * (input_u ** (-1 * self.theta) + input_v ** (-1 * self.theta) - 1) ** (-1/self.theta - 1)

            In_c_du = scipy.integrate.quad(density_function_input_u , sys.float_info.epsilon , 1)[0]

            In_c_dv = scipy.integrate.quad(density_function_input_v, sys.float_info.epsilon, 1)[0]

            marginal_copula_X_given_Y = d_C_d_v/In_c_du

            marginal_copula_Y_given_X = d_C_d_u/In_c_dv


        elif copula == 'Gumbel':

            A_input_u = lambda u : (-1 * np.log(u)) ** self.theta + (-1 * np.log(input_v)) ** self.theta
            A_input_v = lambda v : (-1 * np.log(input_u)) ** self.theta + (-1 * np.log(v)) ** self.theta

            C_input_u = lambda u : np.exp(-1 * (A_input_u(u)) ** 1/self.theta)
            C_input_v = lambda v : np.exp(-1 * (A_input_v(v)) ** 1/self.theta)

            density_function_input_u = lambda u : C_input_u(u) * (u * input_v) ** (-1) * A_input_u(u) ** (-2 + 2/self.theta) * (np.log(u) * np.log(input_v)) ** (self.theta - 1) * (1 + (self.theta - 1) * A_input_u(u) ** (-1/self.theta))

            density_function_input_v = lambda v : C_input_v(v) * (input_u * v) ** (-1) * A_input_v(v) ** (-2 + 2/self.theta) * (np.log(input_u) * np.log(v)) ** (self.theta - 1) * (1 + (self.theta - 1) * A_input_v(v) ** (-1/self.theta))

            A = (-1 * np.log(input_u)) ** self.theta + (-1 * np.log(input_v)) ** self.theta

            C = np.exp(-1 * (A) ** 1 / self.theta)

            d_C_d_v = C * (A) ** ((1-self.theta)/self.theta) * (-1*np.log(input_v)) ** (self.theta - 1) * 1/input_v

            d_C_d_u = C * (A) ** ((1-self.theta)/self.theta) * (-1*np.log(input_u)) ** (self.theta - 1) * 1/input_u

            In_c_du = scipy.integrate.quad(density_function_input_u, sys.float_info.epsilon, 1)[0]

            In_c_dv = scipy.integrate.quad(density_function_input_v, sys.float_info.epsilon, 1)[0]

            marginal_copula_X_given_Y = d_C_d_v/In_c_du

            marginal_copula_Y_given_X = d_C_d_u/In_c_dv

        elif copula == 'Frank':

            density_function_input_u = lambda u: (self.theta * (np.exp(-1 * self.theta) - 1) * (np.exp(-1 * self.theta * (u + input_v))))/(np.exp(-1*self.theta*u - 1) * np.exp(-1*self.theta*input_v - 1) + (np.exp(-1 * self.theta) - 1)) ** 2

            density_function_input_v = lambda v: (self.theta * (np.exp(-1 * self.theta) - 1) * (np.exp(-1 * self.theta * (input_u + v))))/(np.exp(-1*self.theta*input_u - 1) * np.exp(-1*self.theta*v - 1) + (np.exp(-1 * self.theta) - 1)) ** 2

            d_C_d_v = np.exp(-1 * self.theta * input_v) * (np.exp(-1 * self.theta * input_u) -1 )/(np.exp(-1*self.theta)-1 + (np.exp(self.theta*input_u)-1)*(np.exp(self.theta*input_v)-1))

            d_C_d_u = np.exp(-1 * self.theta * input_u) * (np.exp(-1 * self.theta * input_v) -1 )/(np.exp(-1*self.theta)-1 + (np.exp(self.theta*input_u)-1)*(np.exp(self.theta*input_v)-1))

            In_c_du = scipy.integrate.quad(density_function_input_u, sys.float_info.epsilon, 1)[0]

            In_c_dv = scipy.integrate.quad(density_function_input_v, sys.float_info.epsilon, 1)[0]

            marginal_copula_X_given_Y = d_C_d_v/In_c_du

            marginal_copula_Y_given_X = d_C_d_u/In_c_dv

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

            asset_name = []
            asset_log_ret = []
            asset_ECDF = []
            uniformed_transformed_log_ret = []

            for data in self.datas :

                asset_price = []

                for day in range(-1 * (self.params.sample_window) , 1):

                    asset_price.append(data.close[day])

                if sum(np.array(asset_price) < 0.01) > 0:
                    continue

                # Computing the log ret for the sample data
                log_ret = np.log(asset_price[1:]) - np.log(asset_price[:-1])
                # Estimate the Empirical CDF of the log ret
                ECDF = ECDF(asset_log_ret)
                # Transforming the log ret to the uniform variable
                u = ECDF(asset_log_ret)

                # Testing whether the log ret are independent
                test = sm.tsa.stattools.acf(u,nlags = self.params.sample_window/3 ,qstat = True)

                if (test[2][-1] < 0.05):
                    continue

                asset_name.append(data._name)
                asset_log_ret.append(log_ret)
                asset_ECDF.append(ECDF)
                uniformed_transformed_log_ret.append(u)

            for i in range(len(asset_name)-1):
                for j in range(i+1 , len(asset_name)):

                    u = uniformed_transformed_log_ret[i]
                    v = uniformed_transformed_log_ret[j]

                    tau_ = stats.kendalltau(u,v)[0]

                    if tau < tau_:

                        tau = tau_

                        self.selected_pair = [asset_name[i] , asset_name[j]]
                        self.first_asset_sample = asset_log_ret[i]
                        self.second_asset_sample = asset_log_ret[j]
                        self.first_asset_ECDF = asset_ECDF[i]
                        self.second_asset_ECDF = asset_ECDF[j]

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
                self.close(first_asset, exectype=bt.Order.Market)
                self.close(second_asset, exectype=bt.Order.Market)

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































