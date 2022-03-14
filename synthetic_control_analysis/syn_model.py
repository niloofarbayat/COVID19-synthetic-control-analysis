import datetime
import pandas as pd
import numpy as np
from tslib.src.synthcontrol.syntheticControl import RobustSyntheticControl
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib.ticker as ticker
import random
import matplotlib.colors as mcolors
from sklearn import linear_model
from rank_estimation import *
from pyod.models.hbos import HBOS


def mean_error(y_actual, y_pred):
    return np.mean(y_actual - y_pred)

class syn_model(RobustSyntheticControl):


    def __init__(self, state, singVals, dfs, thresh, low_thresh, random_distribution = None, lambdas = [1],
                 mRSC = False, otherStates=[]):

        '''
        Class to perform Robust Synthetic Control with its analysis

        @param
        state:                      (string) the predicted elements
        singVals:                   (int) the number of singular values to retain
        dfs:                        (array) list of dataframe will be used to predict. It will have the same size with lambdas
        thresh:                     (int) The whole time span that will be used to predict
        low_thresh:                 (int) The cut-off between training and testing
        random_distribution:        (function) Add Randomness to the observation
        lambdas:                    (array) The list of weights for each dataframe in dfs. If there is only one dataframe, it will be [1]
        mRSC:                       (bool) If we will perform multi-dimensional Robust Synthetic Control. 
                                           if it is true, dfs should at least contain 2 elements.
        otherStates:                (array) states in the donor pool
        
        '''

        super().__init__(state, singVals, low_thresh, probObservation=1.0, modelType='svd', svdMethod='numpy', otherSeriesKeysArray=otherStates)
        self.state = state
        self.donors = otherStates.copy()
        self.dfs = [df.fillna(0) for df in dfs]
        self.low_thresh = low_thresh
        self.thresh = thresh
        self.mRSC = mRSC
        self.lambdas = lambdas.copy()
        self.random_distribution = random_distribution
        self.actual = self.dfs[0][state]
        self.predictions = None
        self.model_fit = None
        self.train_err = None
        self.test_err = None
        self.denoisedDF = None
        self.train, self.test = self.__split_data()


    def __split_data(self):
        '''
        functions to process and prepare the training and testing data
        '''

        all_rows = self.donors + [self.state]

        df = self.dfs[0][all_rows]

        if not self.mRSC:
            if self.random_distribution:
                trainDF = df + self.random_distribution(df.shape)
                trainDF = trainDF.iloc[:self.low_thresh,:]
            else:
                trainDF = df.iloc[:self.low_thresh,:]
        else:
            num_dimensions = len(self.lambdas)
            trainDF=pd.DataFrame()
            length_one_dimension = df.shape[0]
            for i in range(num_dimensions):
                trainDF=pd.concat([trainDF,self.lambdas[i]*self.dfs[i].iloc[:self.low_thresh,:]], axis=0)

        testDF = df.iloc[self.low_thresh:self.thresh,:]

        return trainDF, testDF



    def fit_model(self, force_positive=True, filter_donor = False, filter_method = 'hbo', backward_donor_eliminate = True, ri_method = "ratio", filter_metrics = mean_squared_error, eps = 1e-4, alpha = 0.05, singval_mathod = 'default', singVals_estimate = False):
        '''
        fit the RobustSyntheticControl model based on given data
        '''

        if filter_donor:
            self.donors = self.filter_donor(filter_metrics, method = filter_method,backward_donor_eliminate = backward_donor_eliminate, eps = eps, alpha = alpha, ri_method = ri_method)[0]

            if len(self.donors) == 0:
                raise ValueError("Donor pool size 0")
            self.otherSeriesKeysArray = self.donors
            self.model.otherSeriesKeysArray = self.donors
            self.train, self.test = self.__split_data()


        if singVals_estimate:
            self.kSingularValues = self.estimate_singVal(method = singval_mathod)
            self.model.kSingularValues = self.kSingularValues
            # print(self.kSingularValues )


        self.fit(self.train)

        denoisedDF = self.model.denoisedDF()

        self.predictions = self.model_predict(force_positive=force_positive)
        self.model_fit = self.model_predict(test = False, force_positive=force_positive)
        self.train_err = self.training_error()
        self.test_err = self.testing_error()
        self.denoisedDF = denoisedDF

    def filter_donor(self, err_metrics, ri_method = "ratio", method = 'hbo', backward_donor_eliminate = True, eps = 1e-4, alpha = 0.05):

        #################################################################################
        ########################### BACKWARD DONOR ELIMINATION ##########################
        #################################################################################

        if backward_donor_eliminate:

            def backward_donor_elimination(rscModel,hi_thresh, low_thresh, metric=mean_squared_error, output = 'error_ratio', shuffle = False):
                '''
                Find the error ratio from removing each of the states within the donorpool

                @param
                metric: metric used to calculate ri values
                '''
                
                def find_lo_high_thresh(rscModel, hi_thresh, low_thresh, donorPool, error_ratio = True):
                    shuffled_df = rscModel.dfs[0].iloc[:hi_thresh,:]
                    if shuffle:
                        shuffled_df = shuffled_df.iloc[np.random.permutation(len(shuffled_df))] 
                    temp_model = syn_model(rscModel.state, rscModel.kSingularValues, [shuffled_df], hi_thresh, low_thresh, 
                                        random_distribution = rscModel.random_distribution, lambdas = rscModel.lambdas, mRSC = rscModel.mRSC, otherStates=donorPool)
                    temp_model.fit_model(force_positive=False)
                    
                    return temp_model.find_ri(metric) if error_ratio else temp_model.train_err

                out_dict = dict()
                    
                for donor in rscModel.donors:
                    donorPool = rscModel.donors.copy()
                    donorPool.remove(donor)
                    #only pre-intervention
                    
                    if output == 'error_ratio': 
                        out_dict[donor] = find_lo_high_thresh(rscModel, hi_thresh, low_thresh,donorPool, error_ratio = True)
                        
                    elif output == 'training_error':
                        out_dict[donor] = find_lo_high_thresh(rscModel, hi_thresh, low_thresh,donorPool, error_ratio = False)
                
                new_donors = np.array(list(out_dict.keys()))
                values = np.array(list(out_dict.values()))
                max_value = max(values)
                new_donors,values = new_donors[(values < max_value)], values[(values < max_value)]
                
                return list(new_donors)


            hi_thresh = self.low_thresh if ri_method == "ratio" else self.thresh
            low_thresh = int(self.low_thresh*.8) if ri_method == "ratio" else self.low_thresh
            

            rscModel1 = syn_model(self.state, self.kSingularValues, self.dfs, hi_thresh, low_thresh, 
                                random_distribution = self.random_distribution, lambdas = self.lambdas, mRSC = self.mRSC, otherStates=self.donors)
            rscModel1.fit_model(filter_donor = False, singVals_estimate = True, singval_mathod ='auto')

            while len(self.donors)>3:
        
                new_donors = backward_donor_elimination(rscModel1, hi_thresh, low_thresh, metric=mean_squared_error, shuffle = False)
                

                rscModel2 =syn_model(self.state, self.kSingularValues, self.dfs, hi_thresh, low_thresh, otherStates=new_donors)
                                
                rscModel2.fit_model(filter_donor = False, singVals_estimate = True, singval_mathod ='auto')

                if  (   ri_method == "ratio" and (rscModel1.find_ri(mean_squared_error) <= rscModel2.find_ri(mean_squared_error)) or  
                        ri_method != "ratio" and (rscModel1.training_error(mean_squared_error) <= rscModel2.training_error(mean_squared_error)) 
                    ):
                    break

                self.donors = new_donors
                rscModel1 = rscModel2
                
        #################################################################################
        #################################################################################
        #################################################################################

        perm_dict = self.permutation_distribution(show_graph = False, include_self = False, ri_method = ri_method, metrics = err_metrics)

        all_donors = np.array(list(perm_dict.keys()))
        values = np.array(list(perm_dict.values()))

        new_donors, new_values = all_donors, values

        if ri_method == "ratio": 
            mu = 1
        else:
            mu = 0

        if method =='hbo':

            train_perm_dict =self.permutation_train_test(include_self = False)
            train_values = np.array(list(train_perm_dict.values()))  ### A bug?

            # for item in train_perm_dict:
            #     print(item, " ", train_perm_dict[item])

            # print("===============================================")
            
            test_perm_dict = self.permutation_train_test(train_err = False, include_self = False)
            test_values = np.array(list(train_perm_dict.values()))

            # for item in test_perm_dict:
            #     print(item, " ", test_perm_dict[item])

            #print(test_perm_dict)

            input_df =  pd.DataFrame([train_values,test_values]).transpose() # pd.DataFrame(values)
            hbos = HBOS(alpha=0.1, contamination=0.15, n_bins=20, tol=0.5)

            #print(input_df)
            hbos.fit(input_df)
            output = hbos.decision_function(input_df)
            res = hbos.predict(input_df)
            res = np.array(res, dtype = bool)

            new_donors = all_donors[~res]
            new_values = test_values[~res]/train_values[~res]
                   

        elif method == 'std':
            std = np.std(values)
            c = mu + 2 * std
            new_donors = all_donors[(values < c) & (values > -c)]
            new_values = values[(values < c) & (values > -c)]

        elif method == 'iqr':
            q75,q25 = np.percentile(values,[75,25])
            intr_qr = q75-q25
            c = q75+(1.5*intr_qr)
            new_donors = all_donors[values < c]
            new_values = values[values < c]

        elif method == 'quantile': 
            q75,q25 = np.percentile(values,[75,25])
            c = q75
            new_donors = all_donors[values < c]
            new_values = values[values < c]

        elif method == 'percentile': 
            q50 = np.percentile(values,[50])
            c = q50
            new_donors = all_donors[values < c]
            new_values = values[values < c]

        elif method == 'bin':
            c = np.histogram(values[values < np.finfo(np.float64).max], bins=10)[1][1]
            new_donors = all_donors[values < c]
            new_values = values[values < c]

        elif method == 'combine':
            c = np.histogram(values[values < np.finfo(np.float64).max], bins=10)[1][1]
            all_donors = all_donors[values < c]
            values = values[values < c]

            std = np.std(values)
            c = mu + 2 * std
            new_donors = all_donors[values < c]
            new_values = values[values < c]

        elif method == 'lasso':
            clf = linear_model.Lasso(alpha=alpha, normalize = True)
            clf.fit(self.train[self.donors],self.train[self.state])
            new_donors = all_donors[abs(clf.coef_) > eps]
            new_values = values[abs(clf.coef_) > eps]
        elif method == 'mcmc':

            donor_to_remove = random.choice(all_donors, weights=values)
            #TODO: continue

        else:
            print('donor selection method is invalid.')

        return list(new_donors), list(new_values)

    def estimate_singVal(self, method = 'default', p = 1):

        X = self.train

        if method == "mRSC":

            a = np.max(X, axis = 0)
            b = np.min(X, axis = 0)
            X = (X - (a + b)/2)/((b-a)/2)
            ########### ROW MEAN
            #mean = np.mean(X, axis = 1)
            #sigma = np.sum(np.square(X[self.state] - mean))/(len(X)-1)
            ###########
            sigma = np.var(X[self.state], ddof = 1) # Column mean
            s = np.linalg.svd(X)[1]
            l = (2.1)* np.sqrt(len(s) * (sigma * p + p * (1-p)))
            h = (3)* np.sqrt(len(s) * (sigma * p + p * (1-p)))

            l = len(s[s > l])
            h = len(s[s > h])

            return (h, l)

        if method == "default":
            return estimate_rank(np.array(X))

        if method == "auto":
            nominal_rank = min(np.array(X).shape)
            
            def find_auto_rank(target, input_df, intervention,otherstates,nominal_rank=30, start = 1):
                nlags = 10
                valid_sv = {}

                for i in range(start,nominal_rank+1):
                    m = syn_model(target, i, input_df, 200, intervention, otherStates=otherstates)
                    m.fit_model(force_positive=False)
                    err = (m.denoisedDF.values - m.train.values)
                    #valid_sv.append([])
                    for j in range(len(err[0])):
                        error = np.array([err[i][j] for i in range(len(err))])
                        error = (error - error.mean()) / error.std()

                        lag_acf, confint, q_stat, p_values = acf(error, nlags=nlags, alpha=.05, qstat = True)
                        lag_pacf, confint = pacf(error, nlags=nlags, alpha=.05)
                        if ((p_values>0.05).all() or i == nominal_rank) and j not in valid_sv: #(p_values.mean()>0.05): #
                            valid_sv[j]= i
                 
                valid_sv = list(valid_sv.values())
                if len(valid_sv):
                    return min(valid_sv) # round((np.array(valid_sv)).mean())
                return 0 #nominal_rank
            
            return find_auto_rank(self.state, self.dfs, self.low_thresh, self.donors, nominal_rank = nominal_rank)




    def model_predict(self, test = True, force_positive = True):

        '''
        do prediction based on the fitted model
        @param
        test: If true, then return the prediction of the model for post-intervention 
        force_positive: If true, turn all the negative prediction to 0

        '''
        if test:
            data = self.test
        else:
            data = self.train

        predictions = np.dot(data[self.donors].fillna(0).values, self.model.weights)
        if force_positive:
            predictions[predictions < 0 ] = 0

        return predictions


    def training_error(self, metrics = mean_squared_error):

        '''
        Find the training error. The metrics is defauted to be mean square error. 
        '''

        return metrics(self.actual[:self.low_thresh], self.model_fit)

    def testing_error(self, metrics = mean_squared_error):
        '''
        Find the testing error. The metrics is defauted to be mean square error. 
        '''

        return metrics(self.actual[self.low_thresh:self.thresh], self.predictions)

    def model_weights(self):

        '''
        Return a dictionary that contain the weights of the model
        '''
        return dict(zip(self.donors, self.model.weights))

    def svd_spectrum(self, show_plot = False, fontsize = 20):

        '''
        Plot the svd_specturm of the model
        '''

        (U, s, Vh) = np.linalg.svd((self.train) - np.mean(self.train))
        s2 = np.power(s, 2)

        if show_plot:
            plt.figure(figsize=(8,6))
            plt.plot(s2)
            plt.grid()
            plt.xlabel("Ordered Singular Values", fontsize=fontsize) 
            plt.ylabel("Energy", fontsize=fontsize)
            plt.title("Singular Value Spectrum", fontsize=fontsize)
            plt.show()

        return U, s, Vh

    def find_ri(self, metrics = mean_squared_error, method = "ratio"):
        '''
        Find the ri score. Defined by the ratio of testing_errror/traing_error
        '''
        if method == "ratio":
            return self.testing_error(metrics)/self.training_error(metrics)
        elif method == "diff":
            return self.testing_error(metrics) - self.training_error(metrics)


    def permutation_train_test(self, train_err = True, include_self = True):
        out_dict = dict()
        #models = dict()

        if include_self:
            if train_err:
                out_dict[self.state] = self.train_err
            else: 
                out_dict[self.state] = self.test_err 

        for donor in self.donors:
            donorPool = self.donors.copy()
            donorPool.remove(donor)
            temp_model = syn_model(donor, self.kSingularValues, self.dfs, self.thresh, self.low_thresh, 
                                random_distribution = self.random_distribution, lambdas = self.lambdas, mRSC = self.mRSC, otherStates=donorPool)
            temp_model.fit_model(force_positive=False)

            # if temp_model.train_err == 0 or temp_model.test_err == 0:
            #     continue

            if train_err:
                out_dict[donor] = temp_model.train_err
            else: 
                out_dict[donor] = temp_model.test_err 

        return out_dict


    def permutation_distribution(self, show_graph = True, include_self = True, show_donors = 10, ax = None, plot_models=0, ri_method = "ratio", metrics=mean_squared_error):
        '''
        Find the premutation_distribution for the states with all it donors

        @param
        show_graph: (bool) True for ploting the permutation_distribution graph
        show_donors: number of donors that will be shown in the graph, from large to small. 
                      Could be 'All' if you want to include all the donors. 
        xes: plt axes object for ploting. Use if you define external plt subplot
        plot_models: number of highest-r_i permutation distribution models to plot
        metric: metric used to calculate ri values
        '''


        out_dict = dict()
        models = dict()

        if include_self:
        
            out_dict[self.state] = self.find_ri(metrics, method = ri_method)
        
        for donor in self.donors:
            donorPool = self.donors.copy()
            donorPool.remove(donor)
            temp_model = syn_model(donor, self.kSingularValues, self.dfs, self.thresh, self.low_thresh, 
                                random_distribution = self.random_distribution, lambdas = self.lambdas, mRSC = self.mRSC, otherStates=donorPool)
            temp_model.fit_model(force_positive=False)
            
            out_dict[donor] = temp_model.find_ri(metrics, method = ri_method)
            models[donor] = temp_model
           
        
        '''
        # possible bug:
        # models are predicting almost the same values, up to a scale factor, uncomment to see mean and stdev
        for i in models.values():
            for j in models.values():
                a = np.append(i.model_fit, i.predictions) / np.append(j.model_fit, j.predictions)
                print("mean: ", np.mean(a), "stdev: ", np.std(a))
        '''
        if show_donors == 'All':
            show_donors = len(out_dict.keys())
        sorted_dict = sorted(out_dict.items(), key=lambda item: item[1])
        if show_graph:
            states = [str(item[0]) for item in sorted_dict[-show_donors:] if item[0] != self.state]
            values = [item[1] for item in sorted_dict[-show_donors:] if item[0] != self.state]

            if include_self:

                states += [str(self.state)]
                values += [out_dict[self.state]]

            if not ax:
                fig, ax = plt.subplots(figsize=(16,6))
            #ax.barh(self.state, out_dict[self.state])
            bar_list = ax.barh(states,values)
            bar_list[-1].set_color('r')
            ax.set_label('RI Score')
            ax.set_title('Permutation distribution graph for %s'%(self.state) )
            for i, v in enumerate(values):
                ax.text(v, i - 0.15, "%2g" % v, fontsize=12)
        
        if plot_models:
            l = [e[0] for e in sorted_dict if e[0] != self.state]
            l.reverse()
            fig, axes = plt.subplots(plot_models, 1, figsize=(15,10*plot_models))
            self.plot(figure=fig, axes=[axes[0]])
            axes[0].set_title(self.state)
            for i in range(1, plot_models):
                models[l[i - 1]].plot(figure=fig, axes=[axes[i]])
                axes[i].set_title(l[i - 1])
        
        return out_dict


    def plot(self, figure = None, axes = [], show_denoise = False, title_text = None, ylimit = None, xlimit = None, logy = False, show_legend = True,
                        show_donors = False, donors_num = None, tick_spacing=30, yaxis = 'Cases', xaxis = 'Date', intervention_date_x_ticks = None, fontsize = 12):

        '''
        Plot the diagram for the model based on its prediction and model fit. 

        @param
        figure: plt figure object for ploting. Used if you define external plt subplot
        axes: list of plt axes object for ploting. Used if you define external plt subplot
        title_test: The thesis for the prediction graph. 
        ylimit: limit for y-axis
        xlimit: limit for x-axis
        logy: if we want to scale of y to be log
        show_donors: if donors will be shown in the graph
        tick_spacing: the spacing in x-axis
        yaxis: the name of the yaxis
        intervention_date_x_ticks: if we want to include the date for the low_thresh in the graph
        fontsize: the fontsize of the text and label in the graph
        donors_num:  number of states that will be included in the donor plot.
        '''
        
        try:
            axesLength = len(axes)
        except TypeError:
            axesLength = 1
            axes = [axes]
        
        if axesLength==0:
            if show_donors:
                figure, axes = plt.subplots(1, 2, figsize=(16,6))
            else:
                figure, axes = plt.subplots(figsize=(16,6))
                axes = [axes]


        if show_donors:
            if not donors_num:
                donors_num = len(self.donors)
            index = np.argsort(np.abs(self.model.weights))[-donors_num:]

            axes[0].barh(np.array(self.donors)[index], (self.model.weights/np.max(self.model.weights))[index], color=list('rgbkymc'))
            axes[0].set_title("Normalized weights for "+str(self.state).replace("-None",""), fontsize=fontsize)
            axes[0].tick_params(axis='both', which='major', labelsize=fontsize)


        ax = axes[-1] if show_donors else axes[0]
        if(ylimit):
            ax.set_ylim(ylimit)
        if(xlimit):
            ax.set_xlim(xlimit)
        if(logy):
            ax.set_yscale('log')
        ax.plot(self.actual.index, self.actual, label='Actuals', color='k', linestyle='-', alpha = 0.7)
        ax.plot(self.test.index, self.predictions, label='Predictions', color='r', linestyle='--')
        ax.plot(self.train.index, self.model_fit, label = 'Fitted model', color='g', linestyle=':')
        if show_denoise:
            ax.plot(self.denoisedDF.index, self.denoisedDF[self.state], label = 'Denoised Model', color='purple', linestyle='-.')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        ax.axvline(x = self.dfs[0].index[self.low_thresh-1], color='k', linestyle='--', linewidth=4)
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        if title_text:
            ax.set_title(title_text+" for "+str(self.state).replace("-None",""), fontsize=fontsize)
        ax.set_xlabel(xaxis, fontsize=fontsize)
        ax.set_ylabel(yaxis, fontsize=fontsize)
        ax.set_xlim(left=0)
        #if show_legend:
        ax.legend(fontsize=25)
        figure.canvas.draw()
        labels = [item.get_text() for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels, rotation=20)
        if intervention_date_x_ticks:
            x_labels = []
            ts = (pd.to_datetime(intervention_date_x_ticks[self.state]))
            for label in labels:
                tmp_date = ts + datetime.timedelta(days = int(label.replace('−','-')))
                x_labels.append(tmp_date.strftime('%Y-%m-%d'))

            ax.set_xticklabels(x_labels, rotation=45)


