import datetime
import re 
import pandas as pd
import numpy as np
from tslib.src.synthcontrol.syntheticControl import RobustSyntheticControl
from matplotlib import pyplot as plt
import hdbscan
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN #For clustering
from sklearn.mixture import GaussianMixture #For GMM clustering
import matplotlib.ticker as ticker
from tslearn.clustering import TimeSeriesKMeans
import random
import matplotlib.colors as mcolors





# function to create filtered dataframe based on thressholds and align timeseries to that start of the spread
def create_filtered_data(df, threshold):
    pattern = re.compile('(Unknown|Unassigned)')
    newdf = pd.DataFrame()
    for location in df.columns:
        if(pattern.search(location)):
            continue
        highnumber = df[df[location].gt(threshold)]
        if(len(highnumber)>0):
            newdf = pd.concat([newdf, pd.DataFrame(columns=[location], data=df.loc[highnumber.index[0]:,location].values)], axis=1)
    return newdf

def create_rolling_data(df, rolling_average_duration = 7, force_monotonicity=True):
    out = df.diff()
    if force_monotonicity:
        out[out < 0] = 0
    out = out.iloc[1:,:].rolling(rolling_average_duration).\
                                mean().iloc[rolling_average_duration:,:]
    return out

#functions to summarized the intervention table based on the given intervention, the output will be used in filter_data_by_intervention
def create_population_adjusted_data(df, population, show_exception = False, county = False):

    new_df = pd.DataFrame()
    # country_total = {}
    exception_list = []
    for state in df:
        try:
            new_df = pd.concat([new_df, 1000000 *df[state]/float(population.loc[state].value)], axis = 1, sort = True)
        except:
            if county:
                exception_list.append(state)
                continue 

            if '-' in state:
                country_name = state[:state.index('-')]
                region_name = state[state.index('-') + 1:]

                if region_name in list(population.index):
                    # Indicate the this is a independent region, collect the region data

                    new_df = pd.concat([new_df, 1000000 *df[state]/float(population.loc[region_name].value)], axis = 1, sort = True)
                else:

                    if country_name not in df.columns:

                        exception_list.append(state)

            else:
                exception_list.append(state)

    if show_exception:

        print('These countries/region do not have population data {}'.format(exception_list))

    return new_df

#Functions to create intervention adjusted data based on the social distancing metrics.

def create_intervention_adjusted_data(df, intervention, rolling_average_duration, ignore_nan=False): 
    intervention_adjusted, intervention_dates = filter_data_by_intervention(df, intervention, ignore_nan=ignore_nan)
    intervention_adjusted_daily = create_rolling_data(intervention_adjusted, rolling_average_duration)
    #intervention_adjusted_daily.index = intervention_adjusted_daily.index-rolling_average_duration
    return intervention_adjusted, intervention_adjusted_daily, intervention_dates


# function to create filtered dataframe based on intervention dates and align timeseries

def filter_data_by_intervention(df, intervention, lag=0, ignore_nan=False):
    newdf = pd.DataFrame()
    if (lag > 0):
        subscript=" -"+str(lag)
    elif (lag < 0):
        subscript=" +"+str(np.abs(lag))
    else:
        subscript=""
    intervention_dates = {}
    for state in df.columns:
        intervention_date = intervention[intervention.name == state].date.values
        if(intervention_date.size>0):
            newdata = df.loc[pd.to_datetime(df.index)>=pd.to_datetime(intervention_date[0])][state].values
            if(not ignore_nan and np.isnan(newdata[:5]).any()):
                print(state)
                continue
            newdf = pd.concat([newdf, pd.DataFrame(columns=[state+subscript],
                                                       data=df.loc[pd.to_datetime(df.index) >= pd.to_datetime(intervention_date[0]) - 
                                                                         datetime.timedelta(days=lag)][state].values)], axis=1)
            intervention_dates[state] = intervention_date[0]
    return newdf, intervention_dates


def get_social_distancing(df, intervention_tried):
    social_distancing = df[['country','name',intervention_tried]].copy(deep=True)
    print(intervention_tried)
    exception_list = []
    for place in social_distancing.name:
    
        try:
            new_value = pd.to_datetime(social_distancing[social_distancing.name == place][intervention_tried],format = '%d.%m.%Y')
            social_distancing.loc[social_distancing.name == place, "date"] = new_value
        except ValueError:
            exception_list.append(place)
    print("Exceptions are", exception_list)
    return social_distancing

def mse(y1, y2):
    return np.sum((y1 - y2) ** 2) / len(y1)/np.sqrt(np.square(y1).sum())

def find_testing_diversion(y1, y2):
    # return np.sum((y1-y2)/y2)
    # y2_copy = y2.copy()
    # y2 = y2[y2_copy != 0]
    # y1 = y1[y2_copy != 0] 
    # return np.sum((y1-y2)/np.abs(y2))

    return np.sum(abs(y1-y2))/np.sum(y2)


def compute_singular_values(df):
    (U, s, Vh) = np.linalg.svd((df) - np.mean(df))
    s2 = np.power(s, 2)
    return pd.Series(s2)

def get_colors(num, picker = 11):
    
    all_c = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),name) for name, color in mcolors.CSS4_COLORS.items())
    selected_c = []
    for c in all_c:
        if (np.array(c[0]) > np.array([0.2, 0.2, 0.2])).all():
            selected_c.append(c[1])
    m = len(selected_c)//num
    return selected_c[picker%m:picker%(len(selected_c) - num * m) + m*num:m]

# function to build and plot synthetic control baswed projections. The first threshold is which regions to use in the donor pool 
# - the ones that have had the timeseries for threshold days and above. The low_thresh is to do predictions for regions that
# have had the spread for at least low_thresh days but below threshold days

# for i in range(20):
#    synth_control_predictions(trial,35,5+i, "Rolling 5-day average deaths data", 2, ylimit=[], savePlots=False, do_only=target, showstates=12,
#                               exclude=exclude1, animation=camera)

def plot_cluster(feature_dict, list_of_dfs, x_labels = [], y_labels = []):
    #plot the images based on the result of cluster_state function
    num_groups = len(feature_dict)
    num_dfs = len(list_of_dfs)
    i = 0
    fig = plt.figure(figsize = (20.0, num_groups*8.0))


    for index in feature_dict:
        group = feature_dict[index]
        ax_list = []
        for j in range(len(list_of_dfs)):
            ax_list.append(fig.add_subplot(num_groups,num_dfs,num_dfs*i+j + 1))
            ax_list[j].xaxis.set_major_locator(ticker.MultipleLocator(15))
            ax_list[j].plot(list_of_dfs[j].index, list_of_dfs[j][group][:])
            ax_list[j].set_xlabel(x_labels[j])
            ax_list[j].set_ylabel(y_labels[j])
            ax_list[j].legend(group)

        i += 1
    plt.show()

def cluster_time_series(time_series_data, cluster_method = 'HDBSCAN', metric = 'euclidean', n_clusters = 4, min_cluster_size = 2, min_sample = 1):
    features = time_series_data.T
    if cluster_method == 'HDBSCAN':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples = min_sample)
        clusterer.fit(features)
        features['cluster']= clusterer.labels_
        feature_dict = features.groupby('cluster').groups
    if cluster_method == 'kmeans':

        #kmeans = KMeans(n_clusters=n_clusters)
        kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, max_iter=5,
                          max_iter_barycenter=5,
                          random_state=0)
        y = kmeans.fit_predict(features)
        features['cluster']= y
        feature_dict = features.groupby('cluster').groups
    if cluster_method == 'AgglomerativeClustering':
        AC = AgglomerativeClustering(n_clusters=4)
        y = AC.fit_predict(features)
        features['cluster']= y
        feature_dict = features.groupby('cluster').groups
    return feature_dict, features


        
def cluster_trend(list_of_dfs, delta, low_thresh, targets, singVals=2, 
                              logy=False, exclude=[], 
                              showstates=4, donorPool=[], mRSC=False, lambdas=[1], error_thresh=1, 
                              random_distribution=None, cluster_method = 'HDBSCAN', n_clusters = 5, cluster_size = 2):
    #cluster states/region/countries based on weights
    weight_features = []
    for target in targets:
        if type(low_thresh) == int:
            low = low_thresh
        else:
            low = low_thresh[target]
  
        newdata = synth_control_predictions(list_of_dfs,low - 7 + delta, low - 7,
                                            "", singVals, ylimit=[], savePlots=False, do_only=[target], showstates=10, donorPool = donorPool,
                               exclude=exclude, svdSpectrum=False, silent=True, showDonors=False, showPlots=False, lambdas=lambdas, mRSC=False, error_thresh = error_thresh)
        weight_features.append(newdata)
        # except ValueError:
        #     print(target)
        #     continue

    feature_list = pd.DataFrame((weight_features))
    feature_list.index=targets
    feature_list.fillna(0, inplace=True)
    #feature_list = feature_list.apply(lambda x: x/x.max(), axis=1)
    feature_columns = feature_list.columns

    if cluster_method == 'HDBSCAN':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size, cluster_selection_method='leaf')
        clustering_labels = clusterer.fit_predict(feature_list[feature_columns])
        feature_list['DB'] = clustering_labels
        feature_dict = feature_list.groupby('DB').groups

    if cluster_method == 'KMEANS':

        kmeans = KMeans(n_clusters=n_clusters)
        y = kmeans.fit_predict(feature_list[feature_columns])
        feature_list.insert((feature_list.shape[1]),'KMeans',y)
        feature_dict = feature_list.groupby('KMeans').groups

    return feature_dict, feature_list







def synth_control_predictions(list_of_dfs, threshold, low_thresh,  title_text, singVals=2, 
                               savePlots=False, ylimit=[], logy=False, exclude=[], 
                               svdSpectrum=False, showDonors=True, do_only=[], showstates=4, animation=[], figure=None, axes=None,
                              donorPool=[], silent=True, showPlots=True, mRSC=False, lambdas=[1], error_thresh=1, yaxis = 'Cases', FONTSIZE = 20, tick_spacing=30, random_distribution=None, check_nan=0):
    
    df = list_of_dfs[0]
    
    if (donorPool):
        otherStates=donorPool.copy()
    else:
        sizes = df.apply(pd.Series.last_valid_index)
        sizes = sizes.fillna(0).astype(int)
        otherStates = list(sizes[sizes>threshold].index)
    if(exclude):
        for member in exclude:
            if(member in otherStates):
                otherStates.remove(member)
    if(do_only):
        for member in exclude:
            if(member in otherStates):
                otherStates.remove(member)
        for member in do_only:
            if(member in otherStates):
                otherStates.remove(member)
            
    
    showstates = np.minimum(showstates,len(otherStates))
    otherStatesNames = otherStates
    otherStatesNames = [w.replace('-None', '') for w in otherStates]
    
    for state in otherStatesNames:
        state.replace("-None","")
    if not silent:
        print(otherStates)
    if(do_only):
        #prediction_states = list(sizes[sizes.index.isin(do_only)].index)
        prediction_states = do_only
        if not silent:
            print(prediction_states)
    else:
        prediction_states = list(sizes[(sizes>low_thresh) & (sizes<=threshold)].index)
    
    if check_nan:
        start = max(df[state].first_valid_index() for state in prediction_states)
        if low_thresh - start > check_nan:
            start = low_thresh - check_nan
        df = df.iloc[start:].reset_index(drop=True)
        list_of_dfs = [df.iloc[start:].reset_index(drop=True) for df in list_of_dfs]
        low_thresh -= start
        otherStates = [state for state in otherStates if df[state].first_valid_index() == 0]
        print('final donorpool: ', otherStates)
    
    for state in prediction_states:
        all_rows = list.copy(otherStates)
        all_rows.append(state)
        if not mRSC:
            if random_distribution:
                trainDF = df + random_distribution(df.shape)
                trainDF = trainDF.iloc[:low_thresh,:]
            else:
                trainDF = df.iloc[:low_thresh,:]
        else:
            num_dimensions = len(lambdas)
            trainDF=pd.DataFrame()
            length_one_dimension = list_of_dfs[0].shape[0]
            for i in range(num_dimensions):
                trainDF=pd.concat([trainDF,lambdas[i]*list_of_dfs[i].iloc[:low_thresh,:]], axis=0)
        if not silent:
            print(trainDF.shape)
        testDF=df.iloc[low_thresh+1:threshold,:]
        rscModel = RobustSyntheticControl(state, singVals, len(trainDF), probObservation=1.0, modelType='svd', svdMethod='numpy', otherSeriesKeysArray=otherStates)
        rscModel.fit(trainDF)
        denoisedDF = rscModel.model.denoisedDF()
        predictions = []
    
        predictions = np.dot(testDF[otherStates].fillna(0).values, rscModel.model.weights)
        predictions_noisy = np.dot(testDF[otherStates].fillna(0).values, rscModel.model.weights)

        predictions[predictions < 0 ] = 0
        x_actual= df[state].index #range(sizes[state])
        actual = df[state] #df.iloc[:sizes[state],:][state]
        
        if (svdSpectrum):
            (U, s, Vh) = np.linalg.svd((trainDF[all_rows]) - np.mean(trainDF[all_rows]))
            s2 = np.power(s, 2)
            plt.figure(figsize=(8,6))
            plt.plot(s2)
            plt.grid()
            plt.xlabel("Ordered Singular Values", fontsize=FONTSIZE) 
            plt.ylabel("Energy", fontsize=FONTSIZE)
            plt.title("Singular Value Spectrum", fontsize=FONTSIZE)
            plt.show()
        x_predictions=df.index[low_thresh:low_thresh+len(predictions)] #range(low_thresh,low_thresh+len(predictions))
        model_fit = np.dot(trainDF[otherStates][:].fillna(0), rscModel.model.weights)

        model_fit[model_fit < 0] = 0
        error = mse(actual[:low_thresh], model_fit)
        if not silent:
            print(state, error)
        # if showPlots:
        #     plt.figure(figsize=(16,6))
        ind = np.argpartition(rscModel.model.weights, -showstates)[-showstates:]
        topstates = [otherStates[i] for i in ind]
        if showDonors:
            axes[0].barh(otherStates, rscModel.model.weights/np.max(rscModel.model.weights), color=list('rgbkymc'))
            axes[0].set_title("Normalized weights for "+str(state).replace("-None",""), fontsize=FONTSIZE)
            axes[0].tick_params(axis='both', which='major', labelsize=FONTSIZE)
        ax = axes[-1] if showDonors else axes
        if(ylimit):
            ax.set_ylim(ylimit)
        if(logy):
            ax.set_yscale('log')
        if(showPlots):
            ax.plot(x_actual,actual,label='Actuals', color='k', linestyle='-')
            ax.plot(x_predictions,predictions,label='Predictions', color='r', linestyle='--')
            #ax.plot(df.index[:low_thresh], model_fit, label = 'Fitted model', color='g', linestyle=':')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

            ax.axvline(x=df.index[low_thresh-1], color='k', linestyle='--', linewidth=4)
            ax.grid()
            ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
            if title_text:
                ax.set_title(title_text+" for "+str(state).replace("-None",""), fontsize=FONTSIZE)
            ax.set_xlabel("Days since Intervention", fontsize=FONTSIZE)
            ax.set_ylabel(yaxis, fontsize=FONTSIZE)
            ax.legend(['Actuals', 'Predictions', 'Fitted Model'], fontsize=FONTSIZE)
            if (savePlots):
                plt.savefig("../Figures/COVID/"+state+'.pdf',bbox_inches='tight')    
            if(animation):
                animation.snap()

                #pred_plot.remove()

            elif(showPlots):
                plt.show()    
    if(error<error_thresh):
        return(dict(zip(otherStates, rscModel.model.weights)))
    else:
        print(state, error)
        return(dict(zip(otherStates, -50*np.ones(len(rscModel.model.weights)))))
# function to track peaks in cases or death rates
def create_peak_clusters(df, threshold=5):
    df_temp = df
    df_cluster = pd.DataFrame(data=df_temp.idxmax(), columns=["days to peak"])
    df_cluster['sizes']=df_temp.apply(pd.Series.last_valid_index)
    df_cluster['peak value'] = df_temp.max()
    df_cluster['initial value'] = df_temp.iloc[0,:]
    df_cluster['sizes'] = df_cluster['sizes'].fillna(0)
    global_peak_size = df_cluster.loc[df_cluster['sizes'] - df_cluster['days to peak'] > threshold]
    #plt.scatter(global_peak_size['days to peak'], (global_peak_size['peak value']), s=2*global_peak_size['initial value']), 
    return global_peak_size



def find_close(df, date_check, infection_level, infection_threshold, county = True, exclude_state = []):
    counties_close = []
    for region in df:
        names = [region]
        if county:
            names = region.split('-')
        if names[0] != 'Unknown' and (names[-1] not in exclude_state) and \
            (infection_level - infection_threshold <= df[region][date_check] <= infection_level + infection_threshold):

            counties_close.append(region)
    return counties_close

# account for the weekends in mobility data
def is_weekend (date):   
    first_weekend = datetime.datetime(2020,1,4)
    try: 
        delta = (date - first_weekend).days
    except: 
        delta = (datetime.datetime.strptime(date,'%Y-%m-%d') - first_weekend).days
    if delta % 7 == 0 or delta % 7 == 1:
        return True
    return False



def find_lockdown_date(state_list,df, mobility_us, max_days = 1, min_mobility = -10, all_populations = None):
    pattern = re.compile('(Unknown|Unassigned)')
    newdf = pd.DataFrame()
    lockdown_dates = {}
    for i in range(1,52):
        if all_populations:
            for j in range(len(all_populations)):
                if state_list[i-1] in all_populations[j]:
                    min_mobility = (j+1)*(-10)
                    break
        count = 0 
        for (date, value) in list(zip(mobility_us[state_list[i-1]].index,mobility_us[state_list[i-1]])):
            if is_weekend(date):
                continue
            if value <= min_mobility:
                count += 1
            elif value >= min_mobility:
                count = 0
            if count == max_days:
                location = state_list[i-1]
                if(pattern.search(location)):
                    continue
                start_date = datetime.datetime.strptime(date,'%Y-%m-%d')
                
                tmp = pd.DataFrame(pd.to_datetime(df[location].index))
                highnumber = df[tmp.gt(start_date)]
                #highnumber2 = df[df[location].gt(0)]
                
                if(len(highnumber)>0):
                    newdf = pd.concat([newdf, pd.DataFrame(columns=[location], data=df.loc[highnumber.index[0]:,location].values)], axis=1)
                    lockdown_dates[location] = start_date
                break

        #print(end_date, mobility_us[state_list[i-1]].index[0], mobility_us[state_list[i-1]].index[-1])
    return newdf, lockdown_dates
        
