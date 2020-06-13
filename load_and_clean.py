import numpy as np
import pandas as pd

# load and clean NYTimes data
def _NYTimes_US():
    # TODO unimplemented
    pass
def _NYTimes_states():
    states = pd.read_csv("../data/covid/NYTimes/us-states.csv")
    cases = states.pivot(index='date', columns='state', values='cases')
    deaths = states.pivot(index='date', columns='state', values='deaths')

    return cases, deaths, states
def _NYTimes_counties():
    counties = pd.read_csv("../data/covid/NYTimes/us-counties.csv")
    counties['county_state'] = counties['county']+'-'+counties['state']
    cases = counties.pivot_table(index='date', columns='county_state', values='cases')
    deaths = counties.pivot_table(index='date', columns='county_state', values='deaths')

    return cases, deaths, counties


# load and clean JHU data
def _JHU_global():
    deaths = pd.read_csv("../data/covid/JHU/time_series_covid19_deaths_global.csv")
    cases = pd.read_csv("../data/covid/JHU/time_series_covid19_confirmed_global.csv")

    deaths['Province-Country'] = deaths['Country/Region']+'-'+deaths['Province/State'].fillna("None")
    deaths['Province-Country'] = deaths['Province-Country'].str.replace("-None", "")
    deaths = deaths.set_index('Province-Country')
    deaths.rename(index={'Georgia':'Georgian Republic'}, inplace=True)
    deaths_after_Jan22 = deaths.loc[:, '1/22/20':].T

    cases['Province-Country'] = cases['Country/Region']+'-'+cases['Province/State'].fillna("None")
    cases['Province-Country'] = cases['Province-Country'].str.replace("-None", "")
    cases = cases.set_index('Province-Country')
    cases.rename(index={'Georgia':'Georgian Republic'}, inplace=True)
    cases_after_Jan22 = cases.loc[:, '1/22/20':].T

    cases_after_Jan22.index = pd.to_datetime(cases_after_Jan22.index, format='%m/%d/%y').strftime('%Y-%m-%d')
    deaths_after_Jan22.index = pd.to_datetime(deaths_after_Jan22.index, format='%m/%d/%y').strftime('%Y-%m-%d')

    return cases_after_Jan22, deaths_after_Jan22

def _JHU_US():
    # TODO unimplemented
    pass


# load and clean mobility data
def _mobility():
    apple = pd.read_csv("../data/mobility/applemobilitytrends-2020-05-30.csv")
    google = pd.read_csv("../data/mobility/Global_Mobility_Report.csv", low_memory=False)

    return apple, google



# load and clean IHME intervention data
def _IHME_intervention():
    sd_data = pd.read_csv("../data/intervention/sdc_sources.csv")
    sd_data['last date'] = 'none'
    for _, row in sd_data.iterrows():
        if row['Mass gathering restrictions'] == "full implementation":
            #print(row['name'])
            row['Mass gathering restrictions'] = row['Stay at Home Order']
        if row['Initial business closures'] == "full implementation":
            #print(row['name'])
            row['Initial business closures'] = row['Non-essential services closed']
        dates_of_intervention = []
        for i in range(3, 7):
            try:
                new_value = pd.to_datetime(row[sd_data.columns[i]], format='%d.%m.%Y')
                dates_of_intervention.append(new_value)
            except ValueError:
                pass

        row['last date'] = np.max(dates_of_intervention)

    return sd_data



# call this with a string appearing in _function_dictionary to specify which dataset to import.
# returns the imported & cleaned dataset.
# if called with a covid dataset, returns both the cases and deaths datasets.
# if called on any NYTimes dataset, additionally returns the raw dataset.
def load_clean(dataset):
    _function_dictionary = {'NYTimes US' : _NYTimes_US,
                        'NYTimes states' : _NYTimes_states,
                        'NYTimes counties' : _NYTimes_counties,
                        'JHU global' : _JHU_global,
                        'JHU US' : _JHU_US,
                        'mobility' : _mobility,
                        'IHME intervention' : _IHME_intervention    
    }

    return _function_dictionary[dataset]()