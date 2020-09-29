import datetime
import pandas as pd
import numpy as np
from tslib.src.synthcontrol.syntheticControl import RobustSyntheticControl
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib.ticker as ticker
import random
import matplotlib.colors as mcolors

    
def mse(y_actual, y_pred):
    return np.sum((y_actual - y_pred) ** 2) / len(y_actual)/np.sqrt(np.square(y_actual).sum())

class syn_model(RobustSyntheticControl):

    def __init__(self, state, singVals, dfs, thresh, low_thresh, random_distribution = None, lambdas = [1],
                 mRSC = False, otherStates=[]):

        super().__init__(state, singVals, low_thresh, probObservation=1.0, modelType='svd', svdMethod='numpy', otherSeriesKeysArray=otherStates) #M needs testing
        self.state = state
        self.donors = otherStates
        self.dfs = [df.fillna(0) for df in dfs]
        self.low_thresh = low_thresh
        self.thresh = thresh
        self.mRSC = mRSC
        self.lambdas = lambdas
        self.random_distribution = random_distribution
        self.train, self.test = self.__split_data(mRSC, random_distribution, lambdas)
        self.actual = self.dfs[0][state]
        self.predictions = None
        self.model_fit = None
        self.train_err = None
        self.test_err = None


    def __split_data(self, mRSC, random_distribution, lambdas):

        all_rows = self.donors + [self.state]
        df = self.dfs[0][all_rows]

        if not mRSC:
            if random_distribution:
                trainDF = df + random_distribution(df.shape)
                trainDF = trainDF.iloc[:self.low_thresh,:]
            else:
                trainDF = df.iloc[:self.low_thresh,:]
        else:
            num_dimensions = len(lambdas)
            trainDF=pd.DataFrame()
            length_one_dimension = df.shape[0]
            for i in range(num_dimensions):
                trainDF=pd.concat([trainDF,lambdas[i]*self.dfs[i].iloc[:self.low_thresh,:]], axis=0)

        testDF = df.iloc[self.low_thresh+1:self.thresh,:]

        return trainDF, testDF



    def fit_model(self, return_denoise = False):

        self.fit(self.train)
        denoisedDF = self.model.denoisedDF()

        self.predictions = self.model_predict()
        self.model_fit = self.model_predict(test = False)
        self.train_err = self.training_error(metrics = mse)
        self.test_err = self.testing_error(metrics = mse)

        if return_denoise:
            return denoisedDF

    def model_predict(self, test = True, force_positive = True):
        if test:
            data = self.test
        else:
            data = self.train

        predictions = np.dot(data[self.donors].fillna(0).values, self.model.weights)
        if force_positive:
            predictions[predictions < 0 ] = 0

        return predictions


    def training_error(self, metrics = mse):

        return metrics(self.actual[:self.low_thresh], self.model_fit)

    def testing_error(self, metrics = mse):

        return metrics(self.actual[self.low_thresh+1:self.thresh], self.predictions)

    def model_weights(self):
        return dict(zip(self.donors, self.model.weights))

    def svd_spectrum(self, FONTSIZE = 20):

        (U, s, Vh) = np.linalg.svd((self.train) - np.mean(self.train))
        s2 = np.power(s, 2)
        plt.figure(figsize=(8,6))
        plt.plot(s2)
        plt.grid()
        plt.xlabel("Ordered Singular Values", fontsize=FONTSIZE) 
        plt.ylabel("Energy", fontsize=FONTSIZE)
        plt.title("Singular Value Spectrum", fontsize=FONTSIZE)
        plt.show()

    def find_ri(self, metrics = mean_squared_error):
        return self.testing_error(metrics)/self.training_error(metrics)

    def permutation_distribution(self, show_graph = True, show_donors = 10):

        out_dict = dict()
        
        out_dict[self.state] = self.find_ri()
        
        for donor in self.donors:
            donorPool = self.donors.copy()
            donorPool.remove(donor)
            temp_model = syn_model(donor, self.kSingularValues, self.dfs, self.thresh, self.low_thresh, 
                                random_distribution = self.random_distribution, lambdas = self.lambdas, mRSC = self.mRSC, otherStates=donorPool)
            temp_model.fit_model()
            out_dict[donor] = temp_model.find_ri()
        if show_graph:
            sorted_dict = sorted(out_dict.items(), key=lambda item: item[1])
            states = [item[0] for item in sorted_dict[-show_donors:]]
            values = [item[1] for item in sorted_dict[-show_donors:]]
            
            fig, ax = plt.subplots(figsize=(16,6))
            ax.barh(states,values)
            ax.set_xlabel('RI Score')
            ax.set_title('Permutation distribution graph')
            for i, v in enumerate(values):
                ax.text(v, i - 0.15, "%3g" % v)
        
        return out_dict


    def plot(self, figure = None, axes = [], title_text = None, ylimit = None, xlimit = None, logy = False, 
                        showDonors = False, tick_spacing=30, yaxis = 'Cases', intervention_date_x_ticks = None, FONTSIZE = 20):
        if len(axes) == 0:
            if showDonors:
                figure, axes = plt.subplots(1, 2, figsize=(16,6))
            else:
                figure, axes = plt.subplots(figsize=(16,6))
                axes = [axes]


        if showDonors:
            axes[0].barh(self.donors, self.model.weights/np.max(self.model.weights), color=list('rgbkymc'))
            axes[0].set_title("Normalized weights for "+str(self.state).replace("-None",""), fontsize=FONTSIZE)
            axes[0].tick_params(axis='both', which='major', labelsize=FONTSIZE)

        ax = axes[-1] if showDonors else axes[0]
        if(ylimit):
            ax.set_ylim(ylimit)
        if(xlimit):
            ax.set_xlim(xlimit)
        if(logy):
            ax.set_yscale('log')
            #if not 
        ax.plot(self.actual.index, self.actual, label='Actuals', color='k', linestyle='-')
        ax.plot(self.test.index, self.predictions, label='Predictions', color='r', linestyle='--')
        ax.plot(self.train.index, self.model_fit, label = 'Fitted model', color='g', linestyle=':')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        ax.axvline(x = self.dfs[0].index[self.low_thresh-1], color='k', linestyle='--', linewidth=4)
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        if title_text:
            ax.set_title(title_text+" for "+str(self.state).replace("-None",""), fontsize=FONTSIZE)
        ax.set_xlabel("Days since intervention", fontsize=FONTSIZE)
        ax.set_ylabel(yaxis, fontsize=FONTSIZE)
        ax.legend(['Actuals', 'Predictions', 'Fitted Model'], fontsize=FONTSIZE)
        figure.canvas.draw()
  
        if intervention_date_x_ticks:
            labels = [item.get_text() for item in ax.get_xticklabels()]
            x_labels = []
            ts = (pd.to_datetime(intervention_date_x_ticks[self.state]))
            #ts = pd.to_datetime(str(date)) 
            for label in labels:
                tmp_date = ts + datetime.timedelta(days = int(label))
                x_labels.append(tmp_date.strftime('%Y-%m-%d'))
            ax.set_xlabel("Date", fontsize=FONTSIZE)

            ax.set_xticklabels(x_labels, rotation=45)