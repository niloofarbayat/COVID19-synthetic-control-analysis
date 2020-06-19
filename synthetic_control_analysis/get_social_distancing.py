import pandas as pd 

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
