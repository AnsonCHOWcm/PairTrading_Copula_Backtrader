import backtrader as bt
import numpy as np
import scipy.stats as stats
import scipy
import sys

class CopulaStrat(bt.Strategy):

    params = (
        ('copula_threshold', 0.05),
    )

    def log(self ,txt):

        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def Copula_AIC(self , copula , theta):

        Likelihood = 0

        u = np.array(self.first_asset_sample)

        v = np.array(self.second_asset_sample)

        # Computing the likelihood of the observed data

        if copula == 'Clayton' :

            Likelihood = sum((theta + 1)*(u ** (-1*theta) + v ** (-1*theta) - 1) ** (-2-1 /theta) * (u) ** (-1 * theta -1) * (v) ** (-1 * theta -1))


        elif copula == 'Gumbel' :

            A = (-1*np.log(u)) ** theta + (-1*np.log(v)) ** theta

            C = np.exp(-1 * (A) ** 1/theta)

            Likelihood = sum(C * (u * v) ** (-1) * A ** (-2 + 2 /theta) * (np.log(u) * np.log(v)) ** (theta -1) * (1 + (theta -1) * A ** (-1/theta)))

        elif copula == 'Frank' :

            Likelihood = sum((theta * (np.exp(-1 * theta) - 1) * (np.exp(-1 * theta * (u + v)))) / (np.exp(-1*theta*u -1) * np.exp(-1*theta*v -1) + (np.exp(-1 * theta) -1 )) ** 2)

        return -2 * Likelihood + 2

    def CopulaFitting(self , tau):

        theta = np.zeros(3)
        highest_AIC = -1*pow(2,31)
        selected_copula = ''
        selected_theta = 0

        # Computing the theta for Clayton Copula

        theta[0] = 2 * tau * (1-tau) ** (-1)

        # Computing the theta for Gumbel Copula

        theta[1] = (1 - tau) ** (-1)

        # Computing the theta for Frank Copula

        intrgral = lambda t : t/(np.exp(t)-1)

        frank_fun = lambda theta : (tau-1)/4 - 1/theta*(1/theta * scipy.integrate.quad(intrgral , sys.float_info.epsilon , theta)[0] - 1)

        theta[2] = scipy.optimize.minimize(frank_fun , 4 , method = 'BFGS' , tol = 1e-5).x

        # Selecting the best fit Copula by AIC

        for i , copula in enumerate(['Clayton' , 'Gumbel' , 'Frank']) :

            AIC = self.Copula_AIC(copula , theta[i])

            if AIC > highest_AIC :

                highest_AIC = AIC

                selected_copula = copula
                selected_theta = theta[i]

        return selected_copula , selected_theta

    def MaginalCopula(self , u , v , copula):

        # Computing marginal copula of the observed data

        if copula == 'Clayton':

            marginal_copula_X_given_Y = v ** (-1 * self.theta -1) * (
                                        u ** (-1 * self.theta) + v ** (-1 * self.theta) -1) ** (-1/self.theta -1)

            marginal_copula_Y_given_X = u ** (-1 * self.theta - 1) * (
                                        u ** (-1 * self.theta) + v ** (-1 * self.theta) - 1) ** (-1 / self.theta - 1)

        elif copula == 'Gumbel':

            A = (-1 * np.log(u)) ** self.theta + (-1 * np.log(v)) ** self.theta

            C = np.exp(-1 * (A) ** 1 / self.theta)

            marginal_copula_X_given_Y = C * (A) ** ((1-self.theta)/self.theta) * (-1*np.log(v)) ** (self.theta -1) * 1/v

            marginal_copula_Y_given_X = C * (A) ** ((1-self.theta)/self.theta) * (-1*np.log(u)) ** (self.theta -1) * 1/u

        elif copula == 'Frank':

            marginal_copula_X_given_Y = np.exp(-1 * self.theta * v) * (np.exp(-1 * self.theta * u) -1) / (np.exp(-1*self.theta)-1 + (np.exp(self.theta*u)-1)*(np.exp(self.theta*v)-1))

            marginal_copula_Y_given_X = np.exp(-1 * self.theta * u) * (np.exp(-1 * self.theta * v) -1) / (np.exp(-1*self.theta)-1 + (np.exp(self.theta*u)-1)*(np.exp(self.theta*v)-1))

        return marginal_copula_X_given_Y , marginal_copula_Y_given_X

    def __init__(self):

        self.rebalance_month = -1
        # Storing the selected pair and its sample log return
        self.selected_pair = []
        self.first_asset_sample = []
        self.second_asset_sample = []
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

        if self.passing_day < 30 :
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

                    for day in range(-31 , 1) :

                        first_asset_price.append(self.datas[i].close[day])
                        second_asset_price.append(self.datas[j].close[day])

                    if sum(np.array(first_asset_price)<0.01) > 0 or sum(np.array(second_asset_price)<0.01) > 0:

                        continue

                    first_asset_log_ret = np.log(first_asset_price[1 : ]) - np.log(first_asset_price[ :-1])

                    second_asset_log_ret = np.log(second_asset_price[1 : ]) - np.log(second_asset_price[ :-1])


                    tau_ = stats.kendalltau(first_asset_log_ret , second_asset_log_ret)[0]

                    if tau < tau_ :

                        tau = tau_

                        self.selected_pair = [self.datas[i]._name , self.datas[j]._name]
                        self.first_asset_sample = first_asset_log_ret
                        self.second_asset_sample = second_asset_log_ret



            self.log('Selected pair %s & %s' %(self.selected_pair[0] , self.selected_pair[1]))

            # Computing the hedge ratio for the selected pair

            self.hedge_ratio = (np.cov(self.first_asset_sample , self.second_asset_sample)[0,0]) / np.var(self.second_asset_sample)

            # Selecting the best fit copula class and fitting its parameter theta

            self.copula , self.theta = self.CopulaFitting(tau)

            # Remove Any Position after the Re-selection if the traded pair is not the selected pair

            for data in self.datas :

                if (not (data._name in self.selected_pair) and self.getposition(data).size != 0) :
                    self.close(data)
                    self.log('Close Position : %s' % data._name)

        # Extracting the CUrrent close price of the selected pair

        for data in self.datas :

            if data._name == self.selected_pair[0] :

                u = data.close[0]
                first_asset = data

            elif data._name == self.selected_pair[1] :

                v = data.close[0]
                second_asset = data

        # Computing the Current Marginal Copula to track the equilibrium status of two asset

        marginal_copula_X_given_Y, marginal_copula_Y_given_X = self.MaginalCopula(u, v, self.copula)

        if not self.position :

            first_asset_proportion = 1 / (1+self.hedge_ratio)

            second_asset_proportion = self.hedge_ratio / (1+self.hedge_ratio)

            # Do Long and Short based on the divergence of the equilibrium of two correlated asset

            if (marginal_copula_X_given_Y < self.params.copula_threshold and marginal_copula_Y_given_X > 1 - self.params.copula_threshold ):

                self.order_target_percent(data = first_asset , target = first_asset_proportion)
                self.order_target_percent(data = second_asset, target = -1 * second_asset_proportion)

                self.log('Long : %s & Short : %s' % (first_asset._name ,second_asset._name))

            elif (marginal_copula_Y_given_X < self.params.copula_threshold and marginal_copula_X_given_Y > 1 - self.params.copula_threshold ) :

                self.order_target_percent(data = first_asset, target = -1*first_asset_proportion)
                self.order_target_percent(data = second_asset, target = second_asset_proportion)

                self.log('Long : %s & Short : %s' % (second_asset._name, first_asset._name))

        else :

            # Close the Position if the divergenece disappear

            if (self.getposition(first_asset) > 0 and (marginal_copula_X_given_Y > self.params.copula_threshold or marginal_copula_Y_given_X < 1 - self.params.copula_threshold) ):
                self.close(first_asset)
                self.close(second_asset)

                self.log('Close Position : Long %s & Short %s' % (first_asset._name ,second_asset._name))

            elif (self.getposition(first_asset) < 0 and (marginal_copula_Y_given_X > self.params.copula_threshold or marginal_copula_X_given_Y < 1 - self.params.copula_threshold) ):

                self.close(first_asset)
                self.close(second_asset)

                self.log('Close Position : Long : %s & Short : %s' % (second_asset._name, first_asset._name))





























