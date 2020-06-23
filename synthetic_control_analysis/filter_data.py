import datetime
import re 
import pandas as pd
import numpy as np
from tslib.src.synthcontrol.syntheticControl import RobustSyntheticControl
from matplotlib import pyplot as plt



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

def create_rolling_data(df, rolling_average_duration = 7):
    return df.diff().iloc[1:,:].rolling(rolling_average_duration).\
                                mean().iloc[rolling_average_duration:,:]

#functions to summarized the intervention table based on the given intervention, the output will be used in filter_data_by_intervention
def create_population_adjusted_data(df, population):

    new_df = pd.DataFrame()

    country_total = {}
    exception_list = []
    for state in df:

        try:
            new_df = pd.concat([new_df, 1000000 *df[state]/float(population[population['Country'] == state].Value)], axis = 1, sort = True)
        except:
            if '-' in state:
                name = state[:state.index('-')]
                if name not in country_total:
                    country_total[name] = df[state]
                else:
                    country_total[name] += df[state] #Collect the data for the countries with province
            else:
                exception_list.append(state)
    for country in country_total: 
        country_total[country].name = country
        new_df = pd.concat([new_df, 1000000 *country_total[country]/float(population[population['Country'] == country].Value)], axis = 1, sort = True)

    return new_df

#Functions to create intervention adjusted data based on the social distancing metrics.

def create_intervention_adjusted_data(df, intervention, rolling_average_duration): 
    intervention_adjusted, intervention_dates = filter_data_by_intervention(df, intervention)
    intervention_adjusted_daily = create_rolling_data(intervention_adjusted, rolling_average_duration)
    intervention_adjusted_daily.index = intervention_adjusted_daily.index-rolling_average_duration
    return intervention_adjusted, intervention_adjusted_daily, intervention_dates


# function to create filtered dataframe based on intervention dates and align timeseries

def filter_data_by_intervention(df, intervention, lag=0):
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
            if(np.isnan(newdata[:5]).any()):
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


# function to build and plot synthetic control baswed projections. The first threshold is which regions to use in the donor pool 
# - the ones that have had the timeseries for threshold days and above. The low_thresh is to do predictions for regions that
# have had the spread for at least low_thresh days but below threshold days

# for i in range(20):
#    synth_control_predictions(trial,35,5+i, "Rolling 5-day average deaths data", 2, ylimit=[], savePlots=False, do_only=target, showstates=12,
#                               exclude=exclude1, animation=camera)
 
def synth_control_predictions(list_of_dfs, threshold, low_thresh,  title_text, singVals=2, 
                               savePlots=False, ylimit=[], logy=False, exclude=[], 
                               svdSpectrum=False, showDonors=True, do_only=[], showstates=4, animation=[], 
                              donorPool=[], silent=True, showPlots=True, mRSC=False, lambdas=[1], error_thresh=1, yaxis = 'Cases', FONTSIZE = 20):
    #print('yo', list_of_dfs,'bo')
    #print(len(list_of_dfs))
    df = list_of_dfs[0]
    sizes = df.apply(pd.Series.last_valid_index)
    sizes = sizes.fillna(0).astype(int)
    
    if (donorPool):
        otherStates=donorPool.copy()
    else:
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
        prediction_states = list(sizes[sizes.index.isin(do_only)].index)
        if not silent:
            print(prediction_states)
    else:
        prediction_states = list(sizes[(sizes>low_thresh) & (sizes<=threshold)].index)
    
    
    for state in prediction_states:
        all_rows = list.copy(otherStates)
        all_rows.append(state)
        if not mRSC:
            trainDF=df.iloc[:low_thresh,:]
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
    
        predictions = np.dot(testDF[otherStates].values, rscModel.model.weights)
        predictions_noisy = np.dot(testDF[otherStates].values, rscModel.model.weights)
        x_actual=range(sizes[state])
        actual = df.iloc[:sizes[state],:][state]
        
        if (svdSpectrum):
            (U, s, Vh) = np.linalg.svd((trainDF[all_rows]) - np.mean(trainDF[all_rows]))
            s2 = np.power(s, 2)
            plt.figure(figsize=(8,6))
            plt.plot(s2)
            plt.grid()
            plt.xlabel("Ordered Singular Values") 
            plt.ylabel("Energy")
            plt.title("Singular Value Spectrum")
            plt.show()
        x_predictions=range(low_thresh,low_thresh+len(predictions))
        model_fit = np.dot(trainDF[otherStates][:], rscModel.model.weights)
        error = mse(actual[:low_thresh], model_fit)
        if not silent:
            print(state, error)
        # if showPlots:
        #     plt.figure(figsize=(16,6))
        ind = np.argpartition(rscModel.model.weights, -showstates)[-showstates:]
        topstates = [otherStates[i] for i in ind]
        if(showDonors):
            donor = plt.subplot(121)        
            donor.barh(otherStates, rscModel.model.weights/np.max(rscModel.model.weights), color=list('rgbkymc'))
            donor.set_title("Normalized weights for "+str(state).replace("-None",""))
            pred = plt.subplot(122)
        
        if(ylimit):
            plt.ylim(ylimit)
        if(logy):
            plt.yscale('log')
        if(showPlots):

            plt.plot(x_actual,actual,label='Actuals', color='k', linestyle='-')
            plt.plot(x_predictions,predictions,label='Predictions', color='r', linestyle='--')
            plt.plot(range(len(model_fit)), model_fit, label = 'Fitted model', color='g', linestyle=':')
            plt.axvline(x=low_thresh-1, color='k', linestyle='--', linewidth=4)
            plt.grid()
            if showDonors:
                plt.title(title_text+" for "+str(state).replace("-None",""))
                plt.xlabel("Days since Intervention")
                plt.ylabel(yaxis)
                plt.legend(['Actuals', 'Predictions', 'Fitted Model'])
            else:
                plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
                plt.title(title_text+" for "+str(state).replace("-None",""), fontsize=FONTSIZE)
                plt.xlabel("Days since Intervention", fontsize=FONTSIZE)
                plt.ylabel(yaxis, fontsize=FONTSIZE)
                plt.legend(['Actuals', 'Predictions', 'Fitted Model'], fontsize=FONTSIZE)
            if (savePlots):
                plt.savefig("../Figures/COVID/"+state+".png")        
            if(animation):
                animation.snap()

                #pred_plot.remove()

            else:
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


def find_lockdown_date(state_list,df, mobility_us, max_days = 1, min_mobility = -10):
    pattern = re.compile('(Unknown|Unassigned)')
    newdf = pd.DataFrame()
    lockdown_dates = {}
    for i in range(1,52):
        count = 0 
        for (date, value) in list(zip(mobility_us[state_list[i-1]].index,mobility_us[state_list[i-1]])):
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
        
