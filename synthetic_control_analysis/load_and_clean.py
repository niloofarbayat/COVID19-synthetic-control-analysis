import os
import sys
import numpy as np
import pandas as pd
import datetime
from subprocess import Popen

import time
import json
import urllib



if sys.platform == 'win32':
    powershell_path = "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe "
else:
    powershell_path = ""
# note that some of these paths are directories, and some are files
_NYTimes_web_path = "https://github.com/nytimes/covid-19-data.git"
_NYTimes_local_path = "../data/covid/NYTimes/"
_NYTimes_mask_data = "../data/covid/NYTimes/mask-use/mask-use-by-county.csv"

_JHU_web_path = "https://github.com/CSSEGISandData/COVID-19.git"
_JHU_local_path = "../data/covid/JHU/"

_google_web_path = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
_google_local_path = "../data/mobility/Global_Mobility_Report.csv"

_apple_web_path = "https://covid19-static.cdn-apple.com/covid19-mobility-data/20%dHotfixDev%d/v3/en-us/applemobilitytrends-%s.csv"
_apple_local_path = "../data/mobility/applemobilitytrends.csv"

_IHME_web_path = None
_IHME_local_path = "../data/intervention/sdc_sources.csv"

_country_pop_local_path = "../data/population/country_pop_WDI.csv"
_county_pop_local_path = "../data/population/co-est2019-annres.csv"

_state_reopen_local_path = "../data/intervention/state_reopen_data.csv"
_temperature_local_path = "../data/temperature/temp_data.csv"

_CTP_US_web_path = "https://api.covidtracking.com/v1/us/daily.csv"
_CTP_US_local_path = "../data/covid/CTP/country.csv"

_CTP_state_web_path = "https://api.covidtracking.com/v1/states/daily.csv"
_CTP_state_local_path = "../data/covid/CTP/state.csv"

_israel_data_web_path = "https://data.gov.il/api/3/action/datastore_search?resource_id=d07c0771-01a8-43b2-96cc-c6154e7fa9bd&records_format=csv"
_israel_data_local_path = "../data/israel/geographic-sum-per-day.csv"

_israel_vaccinations_web_path = "https://data.gov.il/api/3/action/datastore_search?resource_id=12c9045c-1bf4-478a-a9e1-1e876cc2e182&records_format=csv"
_israel_vaccinations_local_path = "../data/israel/vaccinated_city_table.csv"

_israel_population_local_path = "../data/israel/israel_pop.xlsx"




# load and clean NYTimes data
def _import_NYTimes_US():
    country = pd.read_csv(_NYTimes_local_path + "us.csv")[1:]
    return country

def _import_NYTimes_states():
    states = pd.read_csv(_NYTimes_local_path + "us-states.csv")[1:]

    cases = states.pivot(index='date', columns='state', values='cases')
    deaths = states.pivot(index='date', columns='state', values='deaths')

    # for region in regions:
    #     cases[region] = cases[regions[region]].sum(axis = 1)
    #     deaths[region] = deaths[regions[region]].sum(axis = 1)
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

def _import_NYTimes_masks():
    fips = pd.DataFrame.drop_duplicates(pd.read_csv(_NYTimes_local_path + 'us-counties.csv')[['fips', 'county', 'state']])
    fips['county_state'] = fips['county'] + '-' + fips['state']
    fips = fips[['fips', 'county_state']]
    mask = pd.read_csv(_NYTimes_mask_data)
    mask = mask.rename({'COUNTYFP': 'fips'}, axis = 'columns')
    mask = pd.merge(fips, mask, how = 'inner', on = 'fips')[["county_state", "NEVER", "RARELY", "SOMETIMES", "FREQUENTLY", "ALWAYS"]]
    return mask


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

def _import_mobility_apple():
    return pd.read_csv(_apple_local_path)

def _import_mobility_google():
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

    return mobility_data_google, google_social


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

    # i = len(sd_data.index)

    # for region in regions:
    #     info = [np.nan for i in range(len(sd_data.loc[0]))]
    #     info[0] = region
    #     info[1] = 'USA'
        
    #     dates = []
    #     for state in regions[region]:
    #         dates.append(sd_data[sd_data['name'] == state]['last date'].values[0])
    #     info[-1] = np.max(dates)

    #     sd_data.loc[i] = info

    #     i += 1


    return sd_data

#Load and clean the population data
def _import_population_data():

    country_population = pd.read_csv(_country_pop_local_path)
    county_population = pd.read_csv(_county_pop_local_path, header=[3], skipfooter=6)
    new = county_population['Unnamed: 0'].str.strip(".").str.replace(" County","").str.split(pat=",", expand=True)

    county_population['county'] = new[0] +'-' + new[1].str.strip()
    county_population['state']=new[1].str.strip()

    def toInt(s):
        try: return int(s.replace(',',''))
        except: return s
    county_population.columns = county_population.columns.map(toInt)
    county_population[2019] = county_population[2019].map(lambda x: int(x.replace(',','')))
    
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

    # for region in regions:
    #     all_population.loc[region] = all_population.loc[regions[region]].sum()

    
    return all_population, country_population, us_state_population, county_population

def _import_state_reopen_data():
    return pd.read_csv(_state_reopen_local_path, index_col = 0)

def _import_temperature_data():
    _temperature_local_path = "../data/temperature/temp_data.csv"
    temp_data = pd.read_csv(_temperature_local_path)
    temp_data['county_state'] = temp_data['county'] +'-' + temp_data['state']
    out = temp_data.pivot_table(index = "date", columns = "county_state", values = 'avg_temperature').loc['2020-01-22':]
    fips = temp_data[['fips', 'county_state']]
    return out, fips

def _import_CTP_US():
    data = pd.read_csv(_CTP_US_local_path)
    data = data.set_index('date')
    data.index = pd.to_datetime(data.index, format='%Y%m%d').strftime('%Y-%m-%d')
    data = data.iloc[-1::-1]
    # remove deprecated and redundant columns
    stripped = data.drop(labels=['dateChecked', 'deathIncrease', 'hash', 'hospitalized', 'hospitalizedIncrease', 'lastModified', 'negativeIncrease', 'posNeg', 'positiveIncrease', 'total', 'totalTestResultsIncrease'], axis=1)
    return stripped

def _import_CTP_state():
    data = pd.read_csv(_CTP_state_local_path)
    data = data.set_index('date')
    data.index = pd.to_datetime(data.index, format='%Y%m%d').strftime('%Y-%m-%d')
    data = data.iloc[-1::-1]
    # remove deprecated and redundant columns
    stripped = data.drop(labels=['state', 'checkTimeEt', 'commercialScore', 'dataQualityGrade', 'dateChecked', 'dateModified', 'deathIncrease', 'grade', 'hash', 'hospitalized', 'hospitalizedIncrease', 'lastUpdateEt', 'negativeIncrease', 'negativeRegularScore', 'negativeScore', 'posNeg', 'positiveIncrease', 'positiveScore', 'score', 'total', 'totalTestResultsIncrease', 'totalTestResultsSource'], axis=1)
    stats_dict = {stat: stripped.pivot(columns='fips', values=stat) for stat in stripped.columns if stat != 'fips'}
    # convert fips codes into state names
    fips = pd.read_csv(_JHU_local_path + "csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv").set_index('FIPS')
    for _, df in stats_dict.items():
        df.columns = [fips.loc[fips_code]['Province_State'] for fips_code in df.columns]
    return stats_dict

def __israel_pop():
    df_byage = pd.read_csv(_israel_vaccinations_local_path)
    df_all = pd.read_csv(_israel_data_local_path)
    pop = pd.read_excel(_israel_population_local_path)
    pop = pop.loc[pop["Locality Code"].isin(set(pd.unique(df_byage.CityCode)).union(df_all['town_code'])), ["Locality Code", "Total population"]]
    pop.columns = ['code', 'population']
    pop = pop.set_index('code').replace('-', np.nan)
    pop = pop.dropna()
    return pop

def _import_israel_data():
    df_all = pd.read_csv(_israel_data_local_path)
    df_all = df_all.drop(["town"], axis = 1)
    df_all = df_all.groupby(['date', 'town_code']).sum().reset_index().drop('agas_code', axis = 1).replace("<15", "7")
    time_series = {}
    time_series_pop = {}
    pop = __israel_pop()
    for col in df_all.columns[2:]:
        time_series[col] = df_all.pivot(index = 'date', columns = 'town_code', values = col)
        time_series_pop[col] = 1000000 * time_series[col]/pop.loc[time_series[col].columns].population
    return time_series, time_series_pop

def _import_israel_vaccinations():
    df_byage = pd.read_csv(_israel_vaccinations_local_path)
    df = df_byage.set_index(["Date", "CityCode"]).drop(["CityName"], axis = 1).stack().reset_index().replace("<15", "7")

    df.columns = ['Date', 'CityCode', 'level_2', 'value']
    df = df.pivot(index='Date', columns = ["level_2", "CityCode"], values='value').astype(float)
    #df_flat = pd.DataFrame(df.to_records())
    
    pop = __israel_pop()
    df_byage_pop = df_byage.merge(pop.reset_index(), left_on = 'CityCode', right_on = 'code')
    df_byage_pop = df_byage_pop.set_index(["Date", "CityCode"]).drop(["CityName", "code"], axis = 1)
    df_byage_pop = df_byage_pop.replace("<15", "7").astype(float)
    df_byage_pop.iloc[:, :-1] = 100 * (df_byage_pop.iloc[:, :-1].values.T/df_byage_pop.iloc[:, -1].values).T

    df_byage_pop = df_byage_pop.iloc[:, :-1].stack().reset_index()
    df_byage_pop.columns = ['Date', 'CityCode', 'level_2', 'value']
    df_byage_pop = df_byage_pop.pivot(index='Date', columns = ["level_2", "CityCode"], values='value').astype(float)

    #df_byage_pop_flat = pd.DataFrame(df_byage_pop.to_records()).set_index("Date")
    return df, df_byage_pop




# update New York Times data
def _update_NYTimes():
    os.system("git -C %s reset --hard" % _NYTimes_local_path)
    return_value_pull = os.system("git -C %s pull" % _NYTimes_local_path)
    if return_value_pull != 0:
        return_value_clone = os.system("git clone %s %s" % (_NYTimes_web_path, _NYTimes_local_path))
        if return_value_clone != 0:
            print("Unable to update NYTimes data (pull: %d, clone: %d)" % (return_value_pull, return_value_clone), file=sys.stderr)
            return 1
    return 0

# update JHU data
def _update_JHU():
    os.system("git -C %s reset --hard" % _JHU_local_path)
    return_value_pull = os.system("git -C %s pull" % _JHU_local_path)
    if return_value_pull != 0:
        return_value_clone = os.system("git clone %s %s" % (_JHU_web_path, _JHU_local_path))
        if return_value_clone != 0:
            print("Unable to update JHU data (pull: %d, clone: %d)" % (return_value_pull, return_value_clone), file=sys.stderr)
            return 1
    return 0

# update Google mobility data
def _update_google():
    google_hidden_path = "../data/mobility/.Global_Mobility_Report.csv";
    return_value = os.system("curl -o %s -z %s %s" % (google_hidden_path, google_hidden_path, _google_web_path))
    if return_value != 0:
        print("Unable to update Google mobility data (%d)" % return_value, file=sys.stderr)
        return 1
    return_value_copy = os.system(powershell_path + "cp %s %s" % (google_hidden_path, _google_local_path))
    if return_value_copy != 0:
        print("Unable to update Google mobility data (copy: %d)" % return_value_copy, file=sys.stderr)
        return 1
    return 0

# update Apple data
def _update_apple():
    apple_hidden_path = "../data/mobility/.applemobilitytrends.csv"
    try:
        date_last_mod = datetime.date.fromtimestamp(os.path.getmtime(apple_hidden_path))
    except:
        date_last_mod = datetime.date.min

    err = False    
    for day in range(7):
        attempt_date = datetime.date.today() - datetime.timedelta(days=day)
        if attempt_date < date_last_mod:
            os.system(powershell_path + "cp %s %s" % (apple_hidden_path, _apple_local_path))
            os.system(powershell_path + "touch %s" % apple_hidden_path)
            return 0
        # attempt to find url:
        processes = []
        for i in range(14, 14+10):
            for dev in range(20):
                try:
                    processes.append(Popen("curl -s -f -o %s -z %s %s" % (apple_hidden_path, apple_hidden_path, _apple_web_path % (i, dev, attempt_date.strftime('%Y-%m-%d'))), shell=True))
                except OSError:
                    err = True
        if err:
            print("Unable to update Apple mobility data (OSError)", file=sys.stderr)
            for p in processes:
                p.kill()
            break

        if not all(p.wait() for p in processes):
            os.system(powershell_path + "cp %s %s" % (apple_hidden_path, _apple_local_path))
            os.system(powershell_path + "touch %s" % apple_hidden_path)
            return 0

    print("Unable to update Apple mobility data", file=sys.stderr)
    return 1


'''
# OLD APPLE UPDATE CODE:
def _update_apple():
    apple_hidden_path = "../data/mobility/.applemobilitytrends.csv";
    try:
        date_last_mod = datetime.date.fromtimestamp(os.path.getmtime(apple_hidden_path))
    except:
        date_last_mod = datetime.date.min
    for day in range(7):
        attempt_date = datetime.date.today() - datetime.timedelta(days=day)
        if attempt_date <= date_last_mod:
            os.system(powershell_path + "cp %s %s" % (apple_hidden_path, _apple_local_path))
            os.system(powershell_path + "touch %s" % apple_hidden_path)
            return 0
        # attempt to find url:
        for i in range(14, 14+10):
            for dev in range(20):
                if os.system("curl -f -o %s -z %s %s" % (apple_hidden_path, apple_hidden_path, _apple_web_path % (i, dev, attempt_date.strftime('%Y-%m-%d')))) == 0:
                    os.system(powershell_path + "cp %s %s" % (apple_hidden_path, _apple_local_path))
                    os.system(powershell_path + "touch %s" % apple_hidden_path)
                    return 0
    print("Unable to update Apple mobility data", file=sys.stderr)
    return 1
'''






# update IHME data
def _update_IHME():
    return 0 #TODO unimplemented

# update COVID Tracking Project data
def _update_CTP():
    out = 0
    
    US_hidden_path = "../data/covid/CTP/.country.csv"


    return_value_us = os.system("curl -o %s -z %s %s" % (US_hidden_path, US_hidden_path, _CTP_US_web_path))
    if return_value_us != 0:
        print("Unable to update CTP US data (%d)" % return_value_us, file=sys.stderr)
        out += 1
    else:
        return_value_copy_us = os.system(powershell_path + "cp %s %s" % (US_hidden_path, _CTP_US_local_path))
        if return_value_copy_us != 0:
            print("Unable to update CTP US data (copy: %d)" % return_value_copy_us, file=sys.stderr)
            out += 1
    
    state_hidden_path = "../data/covid/CTP/.state.csv"
    return_value_state = os.system("curl -o %s -z %s %s" % (state_hidden_path, state_hidden_path, _CTP_state_web_path))
    if return_value_state != 0:
        print("Unable to update CTP state data (%d)" % return_value_us, file=sys.stderr)
        out += 1
    else:

        return_value_copy_state = os.system(powershell_path + "cp %s %s" % (state_hidden_path, _CTP_state_local_path))
        if return_value_copy_state != 0:
            print("Unable to update CTP state data (copy: %d)" % return_value_copy_state, file=sys.stderr)
            out += 1
    
    return out


def _update_israel():
    data_hidden_path = "../data/israel/.geographic-sum-per-day.csv"
    vaccinations_hidden_path = "../data/israel/.vaccinated_city_table.csv"

    d1 = urllib.request.urlopen(_israel_data_web_path + "&limit=1000000")
    d2 = json.loads(d1.read().decode('utf8'))["result"]
    
    with open(data_hidden_path, "w") as file:
        file.write(",".join([a["id"] for a in d2["fields"]]) + "\n")
        file.write(d2["records"] + "\n")

    d3 = urllib.request.urlopen(_israel_vaccinations_web_path + "&limit=1000000")
    d4 = json.loads(d3.read().decode('utf8'))["result"]
    
    with open(vaccinations_hidden_path, "w") as file:
        file.write(",".join([a["id"] for a in d4["fields"]]) + "\n")
        file.write(d4["records"] + "\n")
        
    os.system(powershell_path + "cp %s %s" % (data_hidden_path, _israel_data_local_path))
    os.system(powershell_path + "cp %s %s" % (vaccinations_hidden_path, _israel_vaccinations_local_path))
    
    return 0
    
    '''
    out = 0
    
    return_value_1 = os.system("curl -o %s -z %s %s" % (data_hidden_path, data_hidden_path, _israel_data_web_path))
    if return_value_1 != 0:
        print("Unable to update Israel town-level data (%d)" % return_value_1, file=sys.stderr)
        out += 1
    else:
        return_value_copy_1 = os.system(powershell_path + "cp %s %s" % (data_hidden_path, _israel_data_local_path))
        if return_value_copy_1 != 0:
            print("Unable to update Israel town-level data (copy: %d)" % return_value_copy_1, file=sys.stderr)
            out += 1
    
    return_value_2 = os.system("curl -o %s -z %s %s" % (vaccinations_hidden_path, vaccinations_hidden_path, _israel_vaccinations_web_path))
    if return_value_2 != 0:
        print("Unable to update Israel vaccination data (%d)" % return_value_2, file=sys.stderr)
        out += 1
    else:
        return_value_copy_2 = os.system(powershell_path + "cp %s %s" % (vaccinations_hidden_path, _israel_vaccinations_local_path))
        if return_value_copy_2 != 0:
            print("Unable to update Israel vaccination data (copy: %d)" % return_value_copy_2, file=sys.stderr)
            out += 1
    '''




def _construct_file_hierarchy():
    def _create_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)
    _create_dir("../data")
    
    _create_dir("../data/covid")
    _create_dir("../data/intervention")
    _create_dir("../data/mobility")
    _create_dir("../data/population")
    _create_dir("../data/temperature")
    _create_dir("../data/israel")
    
    _create_dir("../data/covid/CTP")












# call this with a string appearing in _import_function_dictionary to specify which dataset to import.
# returns the imported & cleaned dataset.
# if called with a covid dataset, returns both the cases and deaths datasets.
# if called on any NYTimes dataset, additionally returns the raw dataset.
def load_clean(dataset):
    _import_function_dictionary = {'NYTimes US' : _import_NYTimes_US,
                        'NYTimes states' : _import_NYTimes_states,
                        'NYTimes counties' : _import_NYTimes_counties,
                        'NYTimes mask': _import_NYTimes_masks,
                        'JHU global' : _import_JHU_global,
                        'JHU US' : _import_JHU_US,
                        'mobility' : _import_mobility,
                        'IHME intervention' : _import_IHME_intervention,  
                        'population': _import_population_data,
                        'state reopen': _import_state_reopen_data,
                        'temperature': _import_temperature_data,
                        'CTP US' : _import_CTP_US,
                        'CTP states' : _import_CTP_state,
                        'mobility Apple' : _import_mobility_apple,
                        'mobility Google' : _import_mobility_google,
                        'Israel geographic' : _import_israel_data,
                        'Israel vaccinations' : _import_israel_vaccinations
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
                                'IHME' : _update_IHME,
                                'CTP' : _update_CTP,
                                'Israel' : _update_israel
    }

    _construct_file_hierarchy()
    
    if dataset:
        return _update_function_dictionary[dataset]()

    out = 0
    for _, f in _update_function_dictionary.items():
        out += f()
    return out
