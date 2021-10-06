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

        testDF = df.iloc[self.low_thresh+1:self.thresh,:]

        return trainDF, testDF



    def fit_model(self, force_positive=True, filter_donor = False, filter_method = 'bin',eps = 1e-4, alpha = 0.05, singval_mathod = 'default', singVals_estimate = False):
        '''
        fit the RobustSyntheticControl model based on given data
        '''
        if singVals_estimate:
            self.kSingularValues = self.estimate_singVal(method = singval_mathod)
            self.model.kSingularValues = self.kSingularValues
            print(self.kSingularValues )

        if filter_donor:
            self.donors = self.filter_donor(method = filter_method, eps = eps, alpha = alpha)[0]

            if len(self.donors) == 0:
                raise ValueError("Donor pool size 0")
            self.otherSeriesKeysArray = self.donors
            self.model.otherSeriesKeysArray = self.donors
            self.train, self.test = self.__split_data()



        self.fit(self.train)

        denoisedDF = self.model.denoisedDF()

        self.predictions = self.model_predict(force_positive=force_positive)
        self.model_fit = self.model_predict(test = False, force_positive=force_positive)
        self.train_err = self.training_error()
        self.test_err = self.testing_error()
        self.denoisedDF = denoisedDF

    def filter_donor(self, method = 'lasso', eps = 1e-4, alpha = 0.05):
        perm_dict = self.permutation_distribution(show_graph = False, include_self = False)

        all_donors = np.array(list(perm_dict.keys()))
        values = np.array(list(perm_dict.values()))

        new_donors, new_values = all_donors, values

        if method == 'std':
            std = np.std(values)
            c = 1 + 2 * std
            new_donors = all_donors[values < c]
            new_values = values[values < c]

        if method == 'bin':
            c = np.histogram(values[values < np.finfo(np.float64).max], bins=10)[1][1]
            new_donors = all_donors[values < c]
            new_values = values[values < c]

        if method == 'combine':
            c = np.histogram(values[values < np.finfo(np.float64).max], bins=10)[1][1]
            all_donors = all_donors[values < c]
            values = values[values < c]

            std = np.std(values)
            c = 1 + 2 * std
            new_donors = all_donors[values < c]
            new_values = values[values < c]

        if method == 'lasso':
            clf = linear_model.Lasso(alpha=alpha, normalize = True)
            clf.fit(self.train[self.donors],self.train[self.state])
            new_donors = all_donors[abs(clf.coef_) > eps]
            new_values = values[abs(clf.coef_) > eps]
       

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
            return find_auto_rank(X, self.low_thresh, nominal_rank)




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

        return metrics(self.actual[self.low_thresh+1:self.thresh], self.predictions)

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

    def find_ri(self, metrics = mean_squared_error):
        '''
        Find the ri score. Defined by the ratio of testing_errror/traing_error
        '''
        return self.testing_error(metrics)/self.training_error(metrics)

    def permutation_distribution(self, show_graph = True, include_self = True, show_donors = 10, ax = None, plot_models=0, metric=mean_squared_error):
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
        
            out_dict[self.state] = self.find_ri(metric)
        
        for donor in self.donors:
            donorPool = self.donors.copy()
            donorPool.remove(donor)
            temp_model = syn_model(donor, self.kSingularValues, self.dfs, self.thresh, self.low_thresh, 
                                random_distribution = self.random_distribution, lambdas = self.lambdas, mRSC = self.mRSC, otherStates=donorPool)
            temp_model.fit_model(force_positive=False)
            
            out_dict[donor] = temp_model.find_ri(metric)
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
        ax.legend(fontsize=fontsize)
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


