import os
import sys
import numpy as np
import pandas as pd





# note that some of these paths are directories, and some are files
_NYTimes_web_path = "https://github.com/nytimes/covid-19-data.git"
_NYTimes_local_path = "../data/covid/NYTimes/"

_JHU_web_path = "https://github.com/CSSEGISandData/COVID-19.git"
_JHU_local_path = "../data/covid/JHU/"

_google_web_path = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
_google_local_path = "../data/mobility/Global_Mobility_Report.csv"

_apple_web_path = "https://covid19-static.cdn-apple.com/covid19-mobility-data/2010HotfixDev18/v3/en-us/applemobilitytrends-2020-06-14.csv"
_apple_local_path = "../data/mobility/applemobilitytrends.csv"

_IHME_web_path = None #TODO unimplemented, line 177
_IHME_local_path = "../data/intervention/sdc_sources.csv"

_country_pop_local_path = "../data/population/country_pop_WDI.xlsx"
_county_pop_local_path = "../data/population/co-est2019-annres.xlsx"

_state_reopen_local_path = "../data/intervention/state_reopen_data.csv"


northeast = ["Connecticut","Maine","Massachusetts","New Hampshire",
             "Rhode Island","Vermont","New Jersey","New York",
             "Pennsylvania", "Delaware","District of Columbia","Maryland"]
midwest = ["Indiana","Illinois","Michigan","Ohio","Wisconsin",
             "Iowa","Kansas","Minnesota","Missouri","Nebraska",
             "North Dakota","South Dakota"]
south = ["Florida","Georgia",
            "North Carolina","South Carolina","Virginia",
            "West Virginia","Alabama","Kentucky","Mississippi",
            "Tennessee","Arkansas","Louisiana","Oklahoma","Texas"]

west= ["Arizona","Colorado","Idaho","New Mexico","Montana",
            "Utah","Nevada","Wyoming","Alaska","California",
            "Hawaii","Oregon","Washington"]

masks_mandated = ['New York', 'Maine', 'Maryland', 'Virginia', 'New Mexico', 'California', 'Michigan', 'Illinois', 'Massachusetts','Delaware', 'Rhode Island']

masks_recommended = ['Montana', 'Idaho', 'Utah', 'Arizona', 'North Dakota', 'South Dakota', 'Kansas', 'Oklahoma', 
                     'Texas', 'North Carolina', 'South Carolina', 'West Virginia', 'Wisconsin','Iowa', 'Missouri', 'Alaska']

eu_countries = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark',
   'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland',
   'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands',
   'Poland', 'Portugal', 'Romania', 'Slovenia', 'Spain', 'Sweden']

regions = {'Northeast': northeast, 
           'Midwest':midwest, 
           'South':south, 
           'West': west}

# load and clean NYTimes data
def _import_NYTimes_US():
    country = pd.read_csv(_NYTimes_local_path + "us.csv")[1:]
    return country

def _import_NYTimes_states():
    states = pd.read_csv(_NYTimes_local_path + "us-states.csv")[1:]


    cases = states.pivot(index='date', columns='state', values='cases')
    deaths = states.pivot(index='date', columns='state', values='deaths')

    for region in regions:
        cases[region] = cases[regions[region]].sum(axis = 1)
        deaths[region] = deaths[regions[region]].sum(axis = 1)
    cases = cases.fillna(0)
    deaths = deaths.fillna(0)


    return cases, deaths, states
def _import_NYTimes_counties():
    counties = pd.read_csv(_NYTimes_local_path + "us-counties.csv")[1:]

    counties['county_state'] = counties['county']+'-'+counties['state']
    cases = counties.pivot_table(index='date', columns='county_state', values='cases')
    deaths = counties.pivot_table(index='date', columns='county_state', values='deaths')
    cases = cases.fillna(0)
    deaths = deaths.fillna(0)

    return cases, deaths, counties


# load and clean JHU data
def _import_JHU_global():
    deaths = pd.read_csv(_JHU_local_path + "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
    cases = pd.read_csv(_JHU_local_path + "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
    country_deaths = deaths.dropna(subset = ['Province/State']).groupby('Country/Region').sum()
    country_cases = cases.dropna(subset = ['Province/State']).groupby('Country/Region').sum()

    country_deaths = country_deaths.loc[:, '1/22/20':].T
    country_cases = country_cases.loc[:, '1/22/20':].T

    deaths['Province-Country'] = deaths['Country/Region']+'-'+deaths['Province/State'].fillna("None")
    deaths['Province-Country'] = deaths['Province-Country'].str.replace("-None", "")
    deaths = deaths.set_index('Province-Country')
    deaths.rename(index={'Georgia':'Georgian Republic'}, inplace=True)
    deaths_after_Jan22 = deaths.loc[:, '1/22/20':].T

    for country in country_deaths.columns:
	    if country not in deaths_after_Jan22.columns:
	        deaths_after_Jan22[country] = country_deaths[country]

    cases['Province-Country'] = cases['Country/Region']+'-'+cases['Province/State'].fillna("None")
    cases['Province-Country'] = cases['Province-Country'].str.replace("-None", "")
    cases = cases.set_index('Province-Country')
    cases.rename(index={'Georgia':'Georgian Republic'}, inplace=True)
    cases_after_Jan22 = cases.loc[:, '1/22/20':].T

    for country in country_cases.columns:
	    if country not in cases_after_Jan22.columns:
	        cases_after_Jan22[country] = country_cases[country]

    cases_after_Jan22.index = pd.to_datetime(cases_after_Jan22.index, format='%m/%d/%y').strftime('%Y-%m-%d')
    deaths_after_Jan22.index = pd.to_datetime(deaths_after_Jan22.index, format='%m/%d/%y').strftime('%Y-%m-%d')

    return cases_after_Jan22, deaths_after_Jan22

def _import_JHU_US():
    deaths = pd.read_csv(_JHU_local_path + "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_us.csv")
    cases = pd.read_csv(_JHU_local_path + "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_us.csv")

    deaths = deaths.set_index('Combined_Key')
    deaths_after_Jan22 = deaths.loc[:, '1/22/20':].T

    cases = cases.set_index('Combined_Key')
    cases_after_Jan22 = cases.loc[:, '1/22/20':].T

    cases_after_Jan22.index = pd.to_datetime(cases_after_Jan22.index, format='%m/%d/%y').strftime('%Y-%m-%d')
    deaths_after_Jan22.index = pd.to_datetime(deaths_after_Jan22.index, format='%m/%d/%y').strftime('%Y-%m-%d')

    return cases_after_Jan22, deaths_after_Jan22


# load and clean mobility data
def _import_mobility():
    mobility_data_apple = pd.read_csv(_apple_local_path)
    mobility_data_google = pd.read_csv(_google_local_path, low_memory=False)

    #mobility_global = mobility_data_google.pivot_table(index='date', values='retail_and_recreation_percent_change_from_baseline', columns='country_region')
    column = list(mobility_data_google.columns).index('retail_and_recreation_percent_change_from_baseline')
    google_work_country = mobility_data_google.pivot_table(index='date', columns='country_region', values='workplaces_percent_change_from_baseline')
    global_google = google_work_country[google_work_country.lt(-30)].apply(pd.Series.first_valid_index)
    google_work_us = mobility_data_google[mobility_data_google.country_region ==  "United States"].pivot_table(index='date', values=mobility_data_google.columns[column], columns='sub_region_1')
    us_google = google_work_us[google_work_us.lt(-30)].apply(pd.Series.first_valid_index)
    google_social = pd.DataFrame(data=pd.concat([global_google, us_google]), columns=['date'])
    google_social['date'] = pd.to_datetime(google_social['date'])
    google_social['name'] = google_social.index

    return mobility_data_apple, mobility_data_google, google_social



# load and clean IHME intervention data
def _import_IHME_intervention():
    sd_data = pd.read_csv(_IHME_local_path)
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

    i = len(sd_data.index)

    for region in regions:
        info = [np.nan for i in range(len(sd_data.loc[0]))]
        info[0] = region
        info[1] = 'USA'
        
        dates = []
        for state in regions[region]:
            dates.append(sd_data[sd_data['name'] == state]['last date'].values[0])
        info[-1] = np.max(dates)

        sd_data.loc[i] = info

        i += 1


    return sd_data

#Load and clean the population data
def _import_population_data():


    country_population = pd.read_excel(_country_pop_local_path)
    county_population = pd.read_excel(_county_pop_local_path, header=[3])
    new = county_population['Unnamed: 0'].str.strip(".").str.replace(" County","").str.split(pat=",", expand=True)

    county_population['county'] = new[0] +'-' + new[1].str.strip()
    county_population['state']=new[1].str.strip()

    county_population = county_population[['county', 2019, 'state']]
    state_population = county_population.groupby('state').sum()
    us_state_population = pd.DataFrame()
    us_state_population['Country'] = state_population.index
    us_state_population['Value'] = state_population[[2019]].values
    county_population = county_population.drop('state', axis = 1).dropna()
    

    us_state_population.columns = ['name', 'value']
    county_population.columns = ['name', 'value']
    country_population.columns = ['name', 'value']

    county_population['name'] = county_population['name'].str.replace(' Parish', '')

    
    us_state_population = us_state_population.set_index('name')
    county_population = county_population.set_index('name')
    country_population = country_population.set_index('name')

    county_population.rename(index={'New York-New York':'New York City-New York'},inplace=True)
    country_population.rename(index = {'Georgia':'Georgian Republic'},inplace=True)

    all_population = pd.concat([country_population, us_state_population, county_population], axis=0, ignore_index=False)

    for region in regions:
        all_population.loc[region] = all_population.loc[regions[region]].sum()

    
    return all_population, country_population, us_state_population, county_population

def _import_state_reopen_data():
    return pd.read_csv(_state_reopen_local_path, index_col = 0)






# update New York Times data
def _update_NYTimes():
    os.system("git -C %s reset --hard" % _NYTimes_local_path)
    if os.system("git -C %s pull" % _NYTimes_local_path) != 0:
        if os.system("git clone %s %s" % (_NYTimes_web_path, _NYTimes_local_path)) != 0:
            print("Unable to update NYTimes data", file=sys.stderr)
            return 1
    return 0

# update JHU data
def _update_JHU():
    os.system("git -C %s reset --hard" % _NYTimes_local_path)
    if os.system("git -C %s pull" % _JHU_local_path) != 0:
        if os.system("git clone %s %s" % (_JHU_web_path, _JHU_local_path)) != 0:
            print("Unable to update JHU data", file=sys.stderr)
            return 1
    return 0

# update Google mobility data
def _update_google():
    google_hidden_path = "../data/mobility/.Global_Mobility_Report.csv";
    if os.system("curl -o %s -z %s %s" % (google_hidden_path, google_hidden_path, _google_web_path)) != 0:
        print("Unable to update Google mobility data", file=sys.stderr)
        return 1
    os.system("cp %s %s" % (google_hidden_path, _google_local_path))
    return 0

# update Apple data
def _update_apple():
    apple_hidden_path = "../data/mobility/.applemobilitytrends.csv";
    if os.system("curl -o %s -z %s %s" % (apple_hidden_path, apple_hidden_path, _apple_web_path)) != 0:
        print("Unable to update Apple mobility data", file=sys.stderr)
        return 1
    os.system("cp %s %s" % (apple_hidden_path, _apple_local_path))
    return 0

# update IHME data
def _update_IHME():
    return 0 #TODO unimplemented









# call this with a string appearing in _import_function_dictionary to specify which dataset to import.
# returns the imported & cleaned dataset.
# if called with a covid dataset, returns both the cases and deaths datasets.
# if called on any NYTimes dataset, additionally returns the raw dataset.
def load_clean(dataset):
    _import_function_dictionary = {'NYTimes US' : _import_NYTimes_US,
                        'NYTimes states' : _import_NYTimes_states,
                        'NYTimes counties' : _import_NYTimes_counties,
                        'JHU global' : _import_JHU_global,
                        'JHU US' : _import_JHU_US,
                        'mobility' : _import_mobility,
                        'IHME intervention' : _import_IHME_intervention,  
                        'population': _import_population_data,
                        'state reopen': _import_state_reopen_data  
    }

    return _import_function_dictionary[dataset]()

# call this with a string appearing in _update_function_dictionary to specify which dataset to update,
# or no arguments to update all datasets.
# returns 0 if successful.
def update_data(dataset=None):
    _update_function_dictionary = {'NYTimes' : _update_NYTimes,
                                'JHU' : _update_JHU,
                                'Google' : _update_google,
                                'Apple' : _update_apple,
                                'IHME' : _update_IHME
    }

    if dataset:
        return _update_function_dictionary[dataset]()

    out = 0
    for _, f in _update_function_dictionary.items():
        out += f()
    return out
