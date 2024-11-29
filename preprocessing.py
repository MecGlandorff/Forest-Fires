



import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(data):

    """ Encodes months like "nov" to numerical values so it can be better handled in heatmaps and the likes, 
        requires pandas dataframe as input and returns pandas dataframe as well"""
    
    # Mappings for months and days
    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    day_mapping = {
        'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4,
        'fri': 5, 'sat': 6, 'sun': 7
    }

    # Map 'month' and 'day' to num values
    data['month'] = data['month'].str.lower().map(month_mapping)
    data['day'] = data['day'].str.lower().map(day_mapping)

    return data

def handle_missing_values(data):
    
    """ Handles missing values by filling them with column means, requires and returns pandas dataframe"""

    return data.fillna(data.mean())

def scale_features(data, features):

    """ Scales specific num values using minmax scaling.
    Input: Data as dataframe and features as list of features that should be scales
    Output: Dataframe with scaled values per features """
    
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    return data

def remove_outliers(data, column):

    """ Remove the outliers from the columns """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
